"""Counterfactual-first Development repair for Project7 V128 Step2.

This module separates two effects that the previous repair accidentally mixed together:

1. the *direct* causal setting -> managed-flow response at the first setting-divergence
   transition, where reference/candidate branches still share the same hydraulic prefix; and
2. the later full-horizon flow difference after hydraulic states have diverged, which is a
   network-feedback trajectory effect and belongs to the joint/autoregressive world model.

The distinction is essential for RTC.  Feeding each branch its own authoritative previous state
and flow makes later candidate-reference flow differences easy to reproduce by persistence even
when the local setting derivative is wrong.  Direct-response supervision below therefore uses a
common authoritative reference prefix for both settings and fails closed unless that prefix is
numerically identical in the stored SWMM counterfactual pair.

Training remains Development-only and uses TrainFit data only.  No Validation/Final/Formal
outcomes, no future online truth, no fabricated conduit-flow labels and no gradient labels are
introduced.  The differentiable TFV gradient remains a downstream diagnostic/solver mechanism;
the training target here is the causal hydraulic trajectory and counterfactual action response.
"""
from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .edge_physics_current_v128 import EdgePhysicsArtifactV128
from .step2_action_identifiable_v128 import (
    ActionIdentifiableActuatorFlowModelV128,
    ActionIdentifiableEdgeAwareSurrogateV128,
    _teacher_step,
)
from .step2_differentiable_v128 import V128SurrogateDesign
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import (
    V127ControlTrainingDesign,
    _candidate_permutation,
    _state_weights,
)
from .step2_train_v127_streaming import (
    _cpu_group,
    _cuda_peak,
    _reset_cuda_peak,
    _select_to_device,
    derive_residual_scales_streaming_v127,
)

COUNTERFACTUAL_FIRST_MODEL_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_EDGE_PHYSICS_SURROGATE_DEV_V2"
)
COUNTERFACTUAL_FIRST_TRAINING_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_DIRECT_THEN_HYDRAULIC_TRAINING_DEV_V2"
)
DIRECT_ACTION_FLOW_SCALE_CONTRACT = (
    "PROJECT7_V128_FIT_ONLY_DIRECT_SAME_PREFIX_ACTION_FLOW_SCALE_V2"
)
DIRECT_ACTION_FLOW_WARMUP_CONTRACT = (
    "PROJECT7_V128_DIRECT_SAME_PREFIX_ACTUATOR_FLOW_WARMUP_DEV_V2"
)
ORACLE_FLOW_TRANSITION_CONTRACT = (
    "PROJECT7_V128_ORACLE_MANAGED_FLOW_HYDRAULIC_TRANSITION_PRETRAIN_DEV_V1"
)
JOINT_DIRECT_STAGE_CONTRACT = (
    "PROJECT7_V128_JOINT_DIRECT_COUNTERFACTUAL_TEACHER_FORCED_DEV_V2"
)

_SETTING_TOL = 1.0e-6
_PREFIX_STATE_ATOL = 2.0e-5
_PREFIX_FLOW_ATOL = 2.0e-5
_DIRECT_EFFECT_FLOOR = 1.0e-4
_DIRECTION_WEIGHT = 0.25


def _finite_float32(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _prefix_context_numpy(
    arrays: dict[str, Any], branch: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    if step == 0:
        state = arrays["initial_state"][branch]
        flow = arrays["previous_actuator_flow"][branch]
    else:
        state = arrays["target_states"][branch][step - 1]
        flow = arrays["target_actuator_flows"][branch][step - 1]
    return _finite_float32(state, label="prefix state"), _finite_float32(flow, label="prefix flow")


def _first_direct_response_spec_numpy(
    arrays: dict[str, Any],
    *,
    reference: int,
    candidate: int,
    require_single_actuator: bool = True,
) -> dict[str, Any] | None:
    """Locate the first action divergence and verify a true same-prefix counterfactual.

    The returned effect is local to that transition.  Later flow differences are deliberately
    excluded because their previous hydraulic states/flows already contain network feedback.
    """
    ref_setting = _finite_float32(arrays["settings"][reference], label="reference settings")
    cand_setting = _finite_float32(arrays["settings"][candidate], label="candidate settings")
    if ref_setting.shape != cand_setting.shape or ref_setting.ndim != 2:
        raise ValueError("counterfactual settings must be matching [H,A] arrays")
    delta = cand_setting - ref_setting
    changed_by_step = np.any(np.abs(delta) > _SETTING_TOL, axis=1)
    steps = np.flatnonzero(changed_by_step)
    if steps.size == 0:
        return None
    step = int(steps[0])
    changed = np.flatnonzero(np.abs(delta[step]) > _SETTING_TOL)
    if require_single_actuator and changed.size != 1:
        return None
    if changed.size == 0:
        return None
    # A first-divergence sample is only a causal direct-response pair if the hydraulic prefix is
    # still common.  Stored SWMM branches are deterministic, but enforce this explicitly so a
    # future cache cannot silently turn feedback into a direct actuator label.
    ref_state, ref_flow = _prefix_context_numpy(arrays, reference, step)
    cand_state, cand_flow = _prefix_context_numpy(arrays, candidate, step)
    state_gap = float(np.max(np.abs(cand_state - ref_state)))
    flow_gap = float(np.max(np.abs(cand_flow - ref_flow)))
    if state_gap > _PREFIX_STATE_ATOL or flow_gap > _PREFIX_FLOW_ATOL:
        return None
    actuator = int(changed[0])
    ref_target = _finite_float32(
        arrays["target_actuator_flows"][reference][step], label="reference target flow"
    )
    cand_target = _finite_float32(
        arrays["target_actuator_flows"][candidate][step], label="candidate target flow"
    )
    return {
        "step": step,
        "actuator_index": actuator,
        "setting_delta": float(delta[step, actuator]),
        "true_flow_delta": float(cand_target[actuator] - ref_target[actuator]),
        "prefix_state_max_abs": state_gap,
        "prefix_flow_max_abs": flow_gap,
    }


def derive_direct_response_scales_v128(
    caches_and_names: Sequence[tuple[Any, Sequence[str]]],
    *,
    sample_rows: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return state scale, temporal-flow scale and direct action-flow scale.

    ``temporal_flow_scale`` represents ordinary q(t)-q(t-1) dynamics.  ``action_flow_scale`` is
    derived independently from first-divergence, same-prefix TrainFit pairs.  They are not
    collapsed into one max-scale: the previous implementation did so and let high-variance
    feedback devices dominate the numerical setting response.
    """
    state_scale, temporal_flow_scale, telemetry = derive_residual_scales_streaming_v127(
        caches_and_names,
        sample_rows=sample_rows,
    )
    temporal = _finite_float32(temporal_flow_scale, label="temporal flow scale").reshape(-1)
    actuator_count = int(temporal.size)
    samples: list[list[float]] = [[] for _ in range(actuator_count)]
    direct_pairs = 0
    zero_direct_pairs = 0
    feedback_samples_excluded = 0
    multi_actuator_excluded = 0
    prefix_mismatch_excluded = 0

    for cache, names in caches_and_names:
        for name in names:
            entry = cache.entry(str(name))
            arrays = entry.arrays
            ref = int(entry.reference_index)
            horizon = int(np.asarray(arrays["settings"][ref]).shape[0])
            for index in entry.indices:
                candidate = int(index)
                if candidate == ref:
                    continue
                ref_setting = np.asarray(arrays["settings"][ref], dtype=np.float32)
                cand_setting = np.asarray(arrays["settings"][candidate], dtype=np.float32)
                delta = cand_setting - ref_setting
                steps = np.flatnonzero(np.any(np.abs(delta) > _SETTING_TOL, axis=1))
                if steps.size == 0:
                    continue
                first = int(steps[0])
                if np.flatnonzero(np.abs(delta[first]) > _SETTING_TOL).size != 1:
                    multi_actuator_excluded += 1
                    feedback_samples_excluded += max(horizon - first, 0)
                    continue
                spec = _first_direct_response_spec_numpy(
                    arrays, reference=ref, candidate=candidate, require_single_actuator=True
                )
                if spec is None:
                    prefix_mismatch_excluded += 1
                    feedback_samples_excluded += max(horizon - first, 0)
                    continue
                direct_pairs += 1
                feedback_samples_excluded += max(horizon - int(spec["step"]) - 1, 0)
                effect = abs(float(spec["true_flow_delta"]))
                if effect > _DIRECT_EFFECT_FLOOR:
                    samples[int(spec["actuator_index"])].append(effect)
                else:
                    zero_direct_pairs += 1

    action_scale = np.full(actuator_count, 1.0e-5, dtype=np.float32)
    action_counts = np.zeros(actuator_count, dtype=np.int64)
    for actuator, values in enumerate(samples):
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        action_scale[actuator] = np.float32(max(float(np.quantile(arr, 0.995)), 1.0e-5))
        action_counts[actuator] = int(arr.size)

    telemetry = dict(telemetry)
    telemetry.update(
        {
            "flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
            "flow_scale_source": "TrainFit first-setting-divergence same-prefix pairs only",
            "temporal_and_action_scales_separate": True,
            "direct_response_pairs": int(direct_pairs),
            "direct_response_nonzero_pairs": int(action_counts.sum()),
            "direct_response_zero_pairs": int(zero_direct_pairs),
            "direct_response_actuators": int(np.sum(action_counts > 0)),
            "feedback_horizon_samples_excluded_from_direct_scale": int(feedback_samples_excluded),
            "multi_actuator_first_divergence_excluded": int(multi_actuator_excluded),
            "prefix_mismatch_excluded": int(prefix_mismatch_excluded),
            "same_prefix_state_atol": _PREFIX_STATE_ATOL,
            "same_prefix_flow_atol": _PREFIX_FLOW_ATOL,
            "holdout_used_for_scale": False,
            "scale_is_engineering_constraint": False,
        }
    )
    return state_scale, temporal.astype(np.float32), action_scale, telemetry


class CounterfactualFirstActuatorFlowModelV128(ActionIdentifiableActuatorFlowModelV128):
    """Separate baseline temporal dynamics from the setting-conditioned response scale."""

    contract = COUNTERFACTUAL_FIRST_MODEL_CONTRACT

    def __init__(self, *args: Any, direct_action_flow_scale: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        direct = torch.as_tensor(direct_action_flow_scale, dtype=torch.float32).reshape(-1)
        if direct.numel() != int(self.actuator_count):
            raise ValueError("direct action-flow scale must match actuator count")
        if not bool(torch.isfinite(direct).all()) or bool((direct <= 0).any()):
            raise ValueError("direct action-flow scale must be positive and finite")
        self.register_buffer("direct_action_flow_scale", direct.clone())

    def set_direct_action_flow_scale(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if value.shape != self.direct_action_flow_scale.shape:
            raise ValueError("direct action-flow scale shape mismatch")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("direct action-flow scale must be positive and finite")
        self.direct_action_flow_scale.copy_(value)

    def forward_prepared(
        self,
        upstream_state: torch.Tensor,
        downstream_state: torch.Tensor,
        setting: torch.Tensor,
        previous_flow: torch.Tensor,
        physics_norm: torch.Tensor,
        identity_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [
            (upstream_state - self.state_mean) / self.state_std,
            (downstream_state - self.state_mean) / self.state_std,
            previous_flow[..., None] / self.flow_std,
            physics_norm,
        ]
        if identity_embedding is not None:
            parts.append(identity_embedding)
        z = self.context_encoder(torch.cat(parts, dim=-1))
        baseline_raw = self.baseline_delta(z).squeeze(-1)
        linear = self.setting_linear_gain(z).squeeze(-1)
        quadratic = self.setting_quadratic_gain(z).squeeze(-1)
        centered = setting * 2.0 - 1.0
        centered_quadratic = centered.square() - (1.0 / 3.0)

        temporal_scale = self.delta_flow_scale.to(device=z.device, dtype=z.dtype)
        action_scale = self.direct_action_flow_scale.to(device=z.device, dtype=z.dtype)
        baseline_delta = torch.tanh(baseline_raw) * temporal_scale
        action_raw = centered * linear + centered_quadratic * quadratic
        action_delta = torch.tanh(action_raw) * action_scale
        predicted_flow = previous_flow + baseline_delta + action_delta
        responsiveness = torch.sigmoid(self.diagnostic_response_logit(z)).squeeze(-1)
        return predicted_flow, responsiveness


class CounterfactualFirstEdgeAwareSurrogateV128(ActionIdentifiableEdgeAwareSurrogateV128):
    contract = COUNTERFACTUAL_FIRST_MODEL_CONTRACT

    def __init__(self, *, direct_action_flow_scale: Any, edge_artifact: EdgePhysicsArtifactV128, **kwargs: Any) -> None:
        super().__init__(edge_artifact=edge_artifact, **kwargs)
        old = self.actuator
        self.actuator = CounterfactualFirstActuatorFlowModelV128(
            state_dim=int(self.transition.state_mean.numel()),
            physics_dim=int(old.physics_mean.numel()),
            hidden_dim=int(self.transition.input.out_features),
            actuator_count=int(old.actuator_count),
            actuator_embedding_dim=int(old.actuator_embedding_dim),
            delta_flow_scale=old.delta_flow_scale.detach().clone(),
            direct_action_flow_scale=direct_action_flow_scale,
        )
        self.v128_contract = COUNTERFACTUAL_FIRST_MODEL_CONTRACT
        self.runtime_metadata.update(
            {
                "counterfactual_first_model_contract": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
                "direct_action_flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
                "gradient_is_training_target": False,
                "development_only": True,
                "full_promotion_allowed": False,
            }
        )


def build_counterfactual_first_v128_model_from_graph(
    graph: Any,
    *,
    edge_artifact: EdgePhysicsArtifactV128,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: Any,
    delta_flow_scale: Any,
    direct_action_flow_scale: Any,
    design: V128SurrogateDesign = V128SurrogateDesign(),
) -> CounterfactualFirstEdgeAwareSurrogateV128:
    edge_artifact.validate(graph)
    if (
        int(getattr(design, "model_step_seconds", -1)),
        int(getattr(design, "control_update_seconds", -1)),
        int(getattr(design, "prediction_horizon_steps", -1)),
        int(getattr(design, "free_control_horizon_steps", -1)),
    ) != (300, 600, 72, 24):
        raise ValueError("counterfactual-first V128 received incompatible time/horizon design")
    return CounterfactualFirstEdgeAwareSurrogateV128(
        edge_artifact=edge_artifact,
        direct_action_flow_scale=torch.as_tensor(direct_action_flow_scale, dtype=torch.float32),
        state_dim=int(state_dim),
        rainfall_dim=int(rainfall_dim),
        node_static_dim=int(np.asarray(graph.static_node_features).shape[1]),
        actuator_physics_dim=int(np.asarray(graph.actuator_physics).shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=int(getattr(design, "hidden_dim", 160)),
        actuator_embedding_dim=int(getattr(design, "actuator_embedding_dim", 16)),
        action_message_dim=int(getattr(design, "action_message_dim", 24)),
        delta_state_scale=torch.as_tensor(delta_state_scale, dtype=torch.float32),
        delta_flow_scale=torch.as_tensor(delta_flow_scale, dtype=torch.float32),
        model_step_seconds=300,
        horizon_steps=72,
        control_update_seconds=600,
        free_control_horizon_steps=24,
        time_contract="PROJECT7_V128_300S_MODEL_600S_RECEDING_CONTROL_V1",
    )


def _direct_effect_loss(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Magnitude + signed-direction loss on already actuator-balanced direct samples."""
    denom = scale.to(device=predicted.device, dtype=predicted.dtype).clamp_min(1.0e-5)
    p = predicted / denom
    t = truth / denom
    magnitude = F.smooth_l1_loss(p, t, beta=0.5)
    informative = torch.abs(truth) > _DIRECT_EFFECT_FLOOR
    if bool(informative.any()):
        # A soft signed margin prevents the magnitude term from matching scale while flipping the
        # local control direction.  This is a counterfactual response loss, not a TFV-gradient label.
        direction = F.softplus(-torch.sign(t[informative]) * p[informative]).mean()
    else:
        direction = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    return magnitude + _DIRECTION_WEIGHT * direction, magnitude, direction


def _cpu_direct_specs(cpu: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    settings = cpu["settings"].detach().cpu().numpy()
    states = cpu["states"].detach().cpu().numpy()
    flows = cpu["flows"].detach().cpu().numpy()
    initial = cpu["initial"].detach().cpu().numpy()
    previous = cpu["previous_flow"].detach().cpu().numpy()
    arrays = {
        "settings": settings,
        "target_states": states,
        "target_actuator_flows": flows,
        "initial_state": initial,
        "previous_actuator_flow": previous,
    }
    specs: list[dict[str, Any]] = []
    for candidate in range(1, int(settings.shape[0])):
        spec = _first_direct_response_spec_numpy(
            arrays, reference=0, candidate=candidate, require_single_actuator=True
        )
        if spec is not None:
            spec = dict(spec)
            spec["candidate_position"] = candidate
            specs.append(spec)
    return specs


def _predict_direct_pair(
    model: Any,
    *,
    cpu: dict[str, torch.Tensor],
    spec: dict[str, Any],
    static: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict ref/candidate next flow from one *shared* authoritative prefix context."""
    k = int(spec["step"])
    candidate = int(spec["candidate_position"])
    actuator = int(spec["actuator_index"])
    if k == 0:
        prev_state = cpu["initial"][0]
        prev_flow = cpu["previous_flow"][0]
    else:
        prev_state = cpu["states"][0, k - 1]
        prev_flow = cpu["flows"][0, k - 1]
    state = prev_state[None].repeat(2, 1, 1).to(device)
    flow = prev_flow[None].repeat(2, 1).to(device)
    setting = torch.stack((cpu["settings"][0, k], cpu["settings"][candidate, k])).to(device)
    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=2)
    q, response = model.actuator.forward_prepared(
        state[:, static["up"]], state[:, static["down"]], setting, flow, physics_norm, identity
    )
    target = torch.stack((cpu["flows"][0, k], cpu["flows"][candidate, k])).to(device)
    return q[:, actuator], target[:, actuator], response, setting


def pretrain_direct_action_flow_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """Cheap A0: absolute flow fit plus causal first-divergence setting response."""
    design.validate()
    torch.manual_seed(design.seed + 2201)
    np.random.seed(design.seed + 2201)
    model.train().to(device)
    static = _static(graph, device)
    optimizer = torch.optim.AdamW(
        model.actuator.parameters(), lr=float(design.learning_rate), weight_decay=float(design.weight_decay)
    )
    absolute_reports: list[float] = []
    direct_reports: list[float] = []
    magnitude_reports: list[float] = []
    direction_reports: list[float] = []
    selected_pairs = 0
    direct_nonzero = 0
    _reset_cuda_peak(device)

    for source, name in _ordered(source_groups, 2201, design.seed):
        cpu = _cpu_group(source_caches[source], name, normalization)
        branches, horizon = cpu["settings"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        absolute = torch.zeros((), device=device)
        # Absolute q fit remains useful, but it is no longer treated as evidence of action
        # identifiability because branch-specific previous flow makes this term easy to shortcut.
        for start in range(0, branches, design.hydraulic_branch_chunk):
            stop = min(start + design.hydraulic_branch_chunk, branches)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
            physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=len(positions))
            term = torch.zeros((), device=device)
            for k in range(horizon):
                prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                q, _ = model.actuator.forward_prepared(
                    prev_state[:, static["up"]], prev_state[:, static["down"]],
                    chunk["settings"][:, k], prev_flow, physics_norm, identity
                )
                error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                term = term + F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.5)
            weighted = (term / float(horizon)) * (float(len(positions)) / float(branches))
            weighted.backward()
            absolute = absolute + weighted.detach()
            del chunk, term, weighted

        specs = _cpu_direct_specs(cpu)
        candidate_budget = min(8 if str(source).upper() == "D2" else 4, len(specs))
        if candidate_budget:
            order = _candidate_permutation(
                len(specs), group_name=name, epoch=2201, seed=design.seed
            )[:candidate_budget]
            losses: list[torch.Tensor] = []
            mags: list[torch.Tensor] = []
            dirs: list[torch.Tensor] = []
            for offset in order.tolist():
                spec = specs[int(offset)]
                q, target, _, _ = _predict_direct_pair(
                    model, cpu=cpu, spec=spec, static=static, device=device
                )
                actuator = int(spec["actuator_index"])
                pred_delta = q[1] - q[0]
                true_delta = target[1] - target[0]
                scale = model.actuator.direct_action_flow_scale[actuator]
                loss, mag, direction = _direct_effect_loss(pred_delta, true_delta, scale)
                losses.append(loss)
                mags.append(mag)
                dirs.append(direction)
                selected_pairs += 1
                direct_nonzero += int(abs(float(true_delta.detach().cpu())) > _DIRECT_EFFECT_FLOOR)
            direct = torch.stack(losses).mean()
            direct.backward()
            direct_reports.append(float(direct.detach()))
            magnitude_reports.append(float(torch.stack(mags).mean().detach()))
            direction_reports.append(float(torch.stack(dirs).mean().detach()))
        torch.nn.utils.clip_grad_norm_(model.actuator.parameters(), design.grad_clip)
        optimizer.step()
        absolute_reports.append(float(absolute))
        del cpu, absolute

    row = {
        "stage": "v128_direct_same_prefix_action_flow_warmup",
        "contract": DIRECT_ACTION_FLOW_WARMUP_CONTRACT,
        "loss": float(np.mean(absolute_reports) + (np.mean(direct_reports) if direct_reports else 0.0)),
        "absolute_flow_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_flow_effect_loss": float(np.mean(direct_reports)) if direct_reports else 0.0,
        "direct_flow_magnitude_loss": float(np.mean(magnitude_reports)) if magnitude_reports else 0.0,
        "direct_flow_direction_loss": float(np.mean(direction_reports)) if direction_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "nonzero_direct_counterfactual_pairs": int(direct_nonzero),
        "common_authoritative_prefix_used": True,
        "full_horizon_feedback_used_as_direct_label": False,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_DIRECT_ACTION_FLOW_A0] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def _oracle_transition_prediction(
    model: Any,
    *,
    prev_state: torch.Tensor,
    previous_flow: torch.Tensor,
    setting: torch.Tensor,
    oracle_flow: torch.Tensor,
    rainfall: torch.Tensor,
    static: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch = int(prev_state.shape[0])
    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=batch)
    with torch.no_grad():
        _, response = model.actuator.forward_prepared(
            prev_state[:, static["up"]], prev_state[:, static["down"]], setting,
            previous_flow, physics_norm, identity
        )
    injection = torch.zeros(batch, prev_state.shape[1], 1, device=prev_state.device, dtype=prev_state.dtype)
    injection = injection.index_add(1, static["up"], -oracle_flow[..., None]).index_add(
        1, static["down"], oracle_flow[..., None]
    )
    action_context = model._typed_action_context(
        state=prev_state,
        setting=setting,
        previous_flow=previous_flow,
        predicted_flow=oracle_flow,
        responsiveness=response.detach(),
        upstream=static["up"],
        downstream=static["down"],
        physics_norm=physics_norm,
        identity_embedding=identity,
    )
    static_norm, edges, inv = model.transition.prepare_static(
        static["static"], static["edges"], batch_size=batch, dtype=prev_state.dtype
    )
    return model.transition.forward_prepared(
        prev_state, rainfall, static_norm, injection, edges, inv, action_context
    )


def pretrain_oracle_flow_transition_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """A1: learn q -> next-state hydraulics with authoritative managed-flow injection."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
    )
    actuator_flags = [p.requires_grad for p in model.actuator.parameters()]
    for p in model.actuator.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(design.learning_rate), weight_decay=float(design.weight_decay))
    absolute_reports: list[float] = []
    direct_state_reports: list[float] = []
    selected_pairs = 0
    _reset_cuda_peak(device)
    try:
        for source, name in _ordered(source_groups, 2301, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute = torch.zeros((), device=device)
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                term = torch.zeros((), device=device)
                for k in range(0, horizon, design.teacher_stride):
                    prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                    prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                    pred = _oracle_transition_prediction(
                        model,
                        prev_state=prev_state,
                        previous_flow=prev_flow,
                        setting=chunk["settings"][:, k],
                        oracle_flow=chunk["flows"][:, k],
                        rainfall=chunk["rainfall"][:, k],
                        static=static,
                    )
                    error = (pred - chunk["states"][:, k]) / model.transition.state_std
                    term = term + F.smooth_l1_loss(
                        error * state_weights, torch.zeros_like(error), beta=0.5
                    )
                steps = len(range(0, horizon, design.teacher_stride))
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                weighted.backward()
                absolute = absolute + weighted.detach()
                del chunk, term, weighted

            specs = _cpu_direct_specs(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _candidate_permutation(len(specs), group_name=name, epoch=2301, seed=design.seed)[:budget]
                state_losses: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    k = int(spec["step"])
                    cand = int(spec["candidate_position"])
                    if k == 0:
                        prev_state_ref = cpu["initial"][0]
                        prev_flow_ref = cpu["previous_flow"][0]
                    else:
                        prev_state_ref = cpu["states"][0, k - 1]
                        prev_flow_ref = cpu["flows"][0, k - 1]
                    prev_state = prev_state_ref[None].repeat(2, 1, 1).to(device)
                    prev_flow = prev_flow_ref[None].repeat(2, 1).to(device)
                    setting = torch.stack((cpu["settings"][0, k], cpu["settings"][cand, k])).to(device)
                    oracle_q = torch.stack((cpu["flows"][0, k], cpu["flows"][cand, k])).to(device)
                    rainfall = cpu["rainfall"][0, k][None].repeat(2, 1).to(device)
                    pred = _oracle_transition_prediction(
                        model, prev_state=prev_state, previous_flow=prev_flow, setting=setting,
                        oracle_flow=oracle_q, rainfall=rainfall, static=static
                    )
                    truth = torch.stack((cpu["states"][0, k], cpu["states"][cand, k])).to(device)
                    pred_delta = (pred[1] - pred[0]) * state_weights
                    true_delta = (truth[1] - truth[0]) * state_weights
                    scale = model.transition.delta_state_scale.to(device=device)
                    state_losses.append(F.smooth_l1_loss(
                        pred_delta / scale.clamp_min(1.0e-5),
                        true_delta / scale.clamp_min(1.0e-5),
                        beta=0.5,
                    ))
                    selected_pairs += 1
                direct_state = torch.stack(state_losses).mean()
                direct_state.backward()
                direct_state_reports.append(float(direct_state.detach()))
            torch.nn.utils.clip_grad_norm_(trainable, design.grad_clip)
            optimizer.step()
            absolute_reports.append(float(absolute))
            del cpu, absolute
    finally:
        for p, flag in zip(model.actuator.parameters(), actuator_flags, strict=True):
            p.requires_grad_(flag)

    row = {
        "stage": "v128_oracle_managed_flow_transition_pretrain",
        "contract": ORACLE_FLOW_TRANSITION_CONTRACT,
        "absolute_state_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_next_state_effect_loss": float(np.mean(direct_state_reports)) if direct_state_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "authoritative_managed_flow_injection": True,
        "actuator_parameters_frozen": True,
        "common_authoritative_prefix_used_for_direct_effect": True,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_ORACLE_FLOW_A1] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_joint_direct_stage_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    """A2: joint one-step fit plus same-prefix direct flow/state response; no feedback shortcut."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay)
    history: list[dict[str, Any]] = []
    _reset_cuda_peak(device)
    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        absolute_reports: list[float] = []
        flow_reports: list[float] = []
        state_reports: list[float] = []
        direction_reports: list[float] = []
        selected_pairs = 0
        for source, name in _ordered(source_groups, epoch + 2400, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute = torch.zeros((), device=device)
            steps = len(range(phase, horizon, design.teacher_stride))
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=len(positions))
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=len(positions), dtype=torch.float32
                )
                term = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    pred_state, q = _teacher_step(
                        model, chunk=chunk, k=k, static=static, physics_norm=physics_norm,
                        identity=identity, static_norm=static_norm, edges=edges, inv=inv
                    )
                    state_error = (pred_state - chunk["states"][:, k]) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    term = term + F.smooth_l1_loss(
                        state_error * state_weights, torch.zeros_like(state_error), beta=0.5
                    ) + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                weighted.backward()
                absolute = absolute + weighted.detach()
                del chunk, term, weighted

            specs = _cpu_direct_specs(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _candidate_permutation(len(specs), group_name=name, epoch=epoch + 2400, seed=design.seed)[:budget]
                direct_terms: list[torch.Tensor] = []
                flow_terms: list[torch.Tensor] = []
                state_terms: list[torch.Tensor] = []
                dir_terms: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    k = int(spec["step"])
                    cand = int(spec["candidate_position"])
                    if k == 0:
                        prev_state_ref = cpu["initial"][0]
                        prev_flow_ref = cpu["previous_flow"][0]
                    else:
                        prev_state_ref = cpu["states"][0, k - 1]
                        prev_flow_ref = cpu["flows"][0, k - 1]
                    prev_state = prev_state_ref[None].repeat(2, 1, 1).to(device)
                    prev_flow = prev_flow_ref[None].repeat(2, 1).to(device)
                    setting = torch.stack((cpu["settings"][0, k], cpu["settings"][cand, k])).to(device)
                    rainfall = cpu["rainfall"][0, k][None].repeat(2, 1).to(device)
                    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=2)
                    static_norm, edges, inv = model.transition.prepare_static(
                        static["static"], static["edges"], batch_size=2, dtype=torch.float32
                    )
                    q, response = model.actuator.forward_prepared(
                        prev_state[:, static["up"]], prev_state[:, static["down"]], setting,
                        prev_flow, physics_norm, identity
                    )
                    injection = torch.zeros(2, prev_state.shape[1], 1, device=device)
                    injection = injection.index_add(1, static["up"], -q[..., None]).index_add(
                        1, static["down"], q[..., None]
                    )
                    action_context = model._typed_action_context(
                        state=prev_state, setting=setting, previous_flow=prev_flow, predicted_flow=q,
                        responsiveness=response, upstream=static["up"], downstream=static["down"],
                        physics_norm=physics_norm, identity_embedding=identity
                    )
                    pred_state = model.transition.forward_prepared(
                        prev_state, rainfall, static_norm, injection, edges, inv, action_context
                    )
                    target_flow = torch.stack((cpu["flows"][0, k], cpu["flows"][cand, k])).to(device)
                    target_state = torch.stack((cpu["states"][0, k], cpu["states"][cand, k])).to(device)
                    actuator = int(spec["actuator_index"])
                    flow_loss, flow_mag, flow_dir = _direct_effect_loss(
                        q[1, actuator] - q[0, actuator],
                        target_flow[1, actuator] - target_flow[0, actuator],
                        model.actuator.direct_action_flow_scale[actuator],
                    )
                    state_loss = F.smooth_l1_loss(
                        ((pred_state[1] - pred_state[0]) * state_weights)
                        / model.transition.delta_state_scale.to(device).clamp_min(1.0e-5),
                        ((target_state[1] - target_state[0]) * state_weights)
                        / model.transition.delta_state_scale.to(device).clamp_min(1.0e-5),
                        beta=0.5,
                    )
                    direct_terms.append(flow_loss + state_loss)
                    flow_terms.append(flow_mag)
                    dir_terms.append(flow_dir)
                    state_terms.append(state_loss)
                    selected_pairs += 1
                direct = torch.stack(direct_terms).mean()
                direct.backward()
                flow_reports.append(float(torch.stack(flow_terms).mean().detach()))
                direction_reports.append(float(torch.stack(dir_terms).mean().detach()))
                state_reports.append(float(torch.stack(state_terms).mean().detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            absolute_reports.append(float(absolute))
            del cpu, absolute

        row = {
            "stage": "v128_counterfactual_first_joint_teacher_forced",
            "contract": JOINT_DIRECT_STAGE_CONTRACT,
            "epoch": int(epoch),
            "teacher_phase": int(phase),
            "absolute_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
            "direct_flow_magnitude_loss": float(np.mean(flow_reports)) if flow_reports else 0.0,
            "direct_flow_direction_loss": float(np.mean(direction_reports)) if direction_reports else 0.0,
            "direct_next_state_effect_loss": float(np.mean(state_reports)) if state_reports else 0.0,
            "selected_direct_counterfactual_pairs": int(selected_pairs),
            "common_authoritative_prefix_used": True,
            "full_horizon_feedback_used_as_direct_label": False,
            "gradient_label_used": False,
            **_cuda_peak(device),
        }
        history.append(row)
        print("[V128_COUNTERFACTUAL_FIRST_A2] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def train_counterfactual_first_hydraulic_stage_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    a0 = pretrain_direct_action_flow_v128(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device, design=design
    )
    a1 = pretrain_oracle_flow_transition_v128(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device,
        depth_index=depth_index, flood_rate_index=flood_rate_index, design=design
    )
    a2 = train_joint_direct_stage_v128(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device,
        depth_index=depth_index, flood_rate_index=flood_rate_index, design=design
    )
    return [a0, a1, *a2]


__all__ = [
    "COUNTERFACTUAL_FIRST_MODEL_CONTRACT",
    "COUNTERFACTUAL_FIRST_TRAINING_CONTRACT",
    "DIRECT_ACTION_FLOW_SCALE_CONTRACT",
    "DIRECT_ACTION_FLOW_WARMUP_CONTRACT",
    "ORACLE_FLOW_TRANSITION_CONTRACT",
    "JOINT_DIRECT_STAGE_CONTRACT",
    "CounterfactualFirstActuatorFlowModelV128",
    "CounterfactualFirstEdgeAwareSurrogateV128",
    "_first_direct_response_spec_numpy",
    "build_counterfactual_first_v128_model_from_graph",
    "derive_direct_response_scales_v128",
    "pretrain_direct_action_flow_v128",
    "pretrain_oracle_flow_transition_v128",
    "train_joint_direct_stage_v128",
    "train_counterfactual_first_hydraulic_stage_v128",
]
