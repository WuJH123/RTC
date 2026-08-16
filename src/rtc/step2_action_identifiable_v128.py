"""Action-identifiable Development upgrade for Project7 V128 Step2.

The current diagnostics show a mixed action-chain failure: actuator-flow directions are often
correct but their magnitudes collapse, while supplying authoritative actuator flows does not
repair the node-hydraulic action effect.  This module addresses both failure modes without
changing the frozen Project7 research objective or using Validation/Final/Formal outcomes.

Key changes are deliberately evidence-driven:

* derive a FIT-only hybrid actuator residual scale from the maximum of the historical temporal
  flow-change scale and the 99.5th percentile candidate-minus-reference flow response;
* make actuator flow explicitly setting-conditioned with a low-order setting basis instead of
  letting previous flow and a learned responsiveness gate suppress the entire action derivative;
* add candidate-minus-reference flow/state effect losses in Stage A and truncated rollout;
* after the exact H360 pairwise objective, run one low-learning-rate Development action-effect
  anchor pass so the TFV objective cannot silently erase hydraulic action sensitivity;
* use the already-audited edge-physics transition for ordinary graph propagation.

The hybrid scale is a numerical surrogate scale, NOT an engineering ramp/rate constraint.
Engineering feasibility remains exclusively in the Step3 decoder/projector contract.
"""
from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .edge_physics_current_v128 import EdgePhysicsArtifactV128
from .models import ActuatorFlowModel
from .step2_differentiable_v128 import V128SurrogateDesign
from .step2_differentiable_v128_edge import EdgeAwareTypedActuatorSurrogateV128
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
from .step2_train_v128_exact import train_objective_stage_streaming_v128

ACTION_IDENTIFIABLE_MODEL_CONTRACT = (
    "PROJECT7_V128_ACTION_IDENTIFIABLE_EDGE_PHYSICS_SURROGATE_DEV_V1"
)
ACTION_IDENTIFIABLE_TRAINING_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_HYDRAULIC_EFFECT_TRAINING_DEV_V1"
)
ACTION_CONDITIONED_FLOW_SCALE_CONTRACT = (
    "PROJECT7_V128_FIT_ONLY_HYBRID_TEMPORAL_ACTION_FLOW_SCALE_V1"
)
POST_OBJECTIVE_ANCHOR_CONTRACT = (
    "PROJECT7_V128_POST_EXACT_ACTION_EFFECT_ANCHOR_DEV_V1"
)


def _finite_float32(value: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")
    return arr


def derive_action_conditioned_residual_scales_v128(
    caches_and_names: Sequence[tuple[Any, Sequence[str]]],
    *,
    sample_rows: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive state scale plus a FIT-only action-aware actuator flow residual scale.

    The historical temporal 0.995 scale remains the lower bound.  For each actuator with
    authoritative TrainFit counterfactual evidence, candidate-minus-reference flow differences
    from branches that actually change that actuator are pooled and their absolute 0.995
    quantile is used as an additional lower bound.  No held-out or audit group can enter this
    function unless the caller incorrectly passes it; the current runner passes FIT groups only.
    """
    state_scale, temporal_flow_scale, telemetry = derive_residual_scales_streaming_v127(
        caches_and_names,
        sample_rows=sample_rows,
    )
    temporal = _finite_float32(temporal_flow_scale, label="temporal flow scale").reshape(-1)
    actuator_count = int(temporal.size)
    samples: list[list[np.ndarray]] = [[] for _ in range(actuator_count)]
    evidence_groups = 0
    evidence_candidates = 0

    for cache, names in caches_and_names:
        for name in names:
            entry = cache.entry(str(name))
            arrays = entry.arrays
            ref = int(entry.reference_index)
            ref_setting = _finite_float32(arrays["settings"][ref], label=f"{name} reference settings")
            ref_flow = _finite_float32(
                arrays["target_actuator_flows"][ref], label=f"{name} reference actuator flow"
            )
            if ref_setting.ndim != 2 or ref_flow.shape != ref_setting.shape:
                raise ValueError(f"{name}: settings/actuator-flow HxA schema mismatch")
            if ref_setting.shape[1] != actuator_count:
                raise ValueError(f"{name}: actuator count differs from temporal scale")
            group_has_evidence = False
            for index in entry.indices:
                if int(index) == ref:
                    continue
                cand_setting = _finite_float32(
                    arrays["settings"][index], label=f"{name} candidate settings"
                )
                cand_flow = _finite_float32(
                    arrays["target_actuator_flows"][index], label=f"{name} candidate actuator flow"
                )
                if cand_setting.shape != ref_setting.shape or cand_flow.shape != ref_flow.shape:
                    raise ValueError(f"{name}: candidate setting/flow schema mismatch")
                changed = np.any(np.abs(cand_setting - ref_setting) > 1.0e-6, axis=0)
                changed_indices = np.flatnonzero(changed)
                if changed_indices.size == 0:
                    continue
                group_has_evidence = True
                evidence_candidates += 1
                abs_delta = np.abs(cand_flow - ref_flow)
                for actuator_index in changed_indices.tolist():
                    values = abs_delta[:, int(actuator_index)]
                    positive = values[np.isfinite(values) & (values > 0.0)]
                    if positive.size:
                        samples[int(actuator_index)].append(positive.astype(np.float32, copy=False))
            evidence_groups += int(group_has_evidence)

    action_scale = np.zeros(actuator_count, dtype=np.float32)
    action_sample_count = np.zeros(actuator_count, dtype=np.int64)
    for actuator_index, values in enumerate(samples):
        if not values:
            continue
        pooled = np.concatenate(values).astype(np.float32, copy=False)
        if pooled.size:
            action_scale[actuator_index] = np.float32(np.quantile(pooled, 0.995))
            action_sample_count[actuator_index] = int(pooled.size)

    hybrid = np.maximum(temporal, action_scale).clip(min=1.0e-5).astype(np.float32)
    positive = action_scale > 0.0
    upgraded = action_scale > temporal
    ratios = np.divide(
        temporal,
        action_scale,
        out=np.full_like(temporal, np.nan, dtype=np.float32),
        where=positive,
    )
    telemetry = dict(telemetry)
    telemetry.update(
        {
            "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
            "flow_scale_source": "TrainFit groups supplied to the current runner only",
            "fit_action_evidence_groups": int(evidence_groups),
            "fit_action_evidence_candidates": int(evidence_candidates),
            "fit_action_evidence_actuators": int(np.sum(positive)),
            "hybrid_scale_upgraded_actuators": int(np.sum(upgraded)),
            "temporal_to_action_ratio_median_positive": (
                float(np.nanmedian(ratios)) if np.any(positive) else float("nan")
            ),
            "action_sample_count_total": int(action_sample_count.sum()),
            "holdout_used_for_scale": False,
            "scale_is_engineering_constraint": False,
        }
    )
    return state_scale, hybrid, telemetry


class ActionIdentifiableActuatorFlowModelV128(ActuatorFlowModel):
    """Actuator model with an explicit, non-gated setting derivative.

    Previous flow remains a causal feature and the prediction remains residual in q, but the
    setting pathway is factorized into linear/quadratic basis terms.  Unlike the inherited V127
    model, a learned sigmoid ``responsiveness`` cannot multiply the entire flow delta to nearly
    zero.  The returned responsiveness is retained only as a learned diagnostic/context feature.
    """

    contract = ACTION_IDENTIFIABLE_MODEL_CONTRACT

    def __init__(
        self,
        state_dim: int,
        physics_dim: int,
        hidden_dim: int = 160,
        *,
        actuator_count: int,
        actuator_embedding_dim: int = 16,
        delta_flow_scale: torch.Tensor | np.ndarray,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            physics_dim=physics_dim,
            hidden_dim=hidden_dim,
            actuator_count=actuator_count,
            actuator_embedding_dim=actuator_embedding_dim,
            bounded_flow_residual=True,
            delta_flow_scale=torch.as_tensor(delta_flow_scale, dtype=torch.float32),
        )
        # Remove the inherited setting-entangled residual heads.  Keeping them as unused
        # parameters would make source/checkpoint semantics ambiguous and waste optimizer work.
        del self.encoder
        del self.response_logit
        del self.flow_delta
        context_dim = state_dim * 2 + physics_dim + 1 + self.actuator_embedding_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.baseline_delta = nn.Linear(hidden_dim, 1)
        self.setting_linear_gain = nn.Linear(hidden_dim, 1)
        self.setting_quadratic_gain = nn.Linear(hidden_dim, 1)
        self.diagnostic_response_logit = nn.Linear(hidden_dim, 1)

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
        baseline = self.baseline_delta(z).squeeze(-1)
        linear = self.setting_linear_gain(z).squeeze(-1)
        quadratic = self.setting_quadratic_gain(z).squeeze(-1)
        centered = setting * 2.0 - 1.0
        centered_quadratic = centered.square() - (1.0 / 3.0)
        raw_delta = baseline + centered * linear + centered_quadratic * quadratic
        scale = self.delta_flow_scale.to(device=raw_delta.device, dtype=raw_delta.dtype)
        delta = torch.tanh(raw_delta) * scale
        predicted_flow = previous_flow + delta
        responsiveness = torch.sigmoid(self.diagnostic_response_logit(z)).squeeze(-1)
        return predicted_flow, responsiveness


def _response_weighted_effect_loss(
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Response-normalized Smooth-L1 that does not let sparse action effects disappear."""
    target_scale = scale.to(device=predicted_delta.device, dtype=predicted_delta.dtype).clamp_min(1.0e-5)
    while target_scale.ndim < predicted_delta.ndim:
        target_scale = target_scale.unsqueeze(0)
    normalized_pred = predicted_delta / target_scale
    normalized_true = true_delta / target_scale
    raw = F.smooth_l1_loss(normalized_pred, normalized_true, beta=0.5, reduction="none")
    weight = 1.0 + torch.abs(normalized_true)
    weight = weight / weight.mean().clamp_min(1.0e-6)
    return (raw * weight).mean()


class ActionIdentifiableEdgeAwareSurrogateV128(EdgeAwareTypedActuatorSurrogateV128):
    contract = ACTION_IDENTIFIABLE_MODEL_CONTRACT

    def __init__(self, *, edge_artifact: EdgePhysicsArtifactV128, **kwargs: Any) -> None:
        super().__init__(edge_artifact=edge_artifact, **kwargs)
        old = self.actuator
        self.actuator = ActionIdentifiableActuatorFlowModelV128(
            state_dim=int(self.transition.state_mean.numel()),
            physics_dim=int(old.physics_mean.numel()),
            hidden_dim=int(self.transition.input.out_features),
            actuator_count=int(old.actuator_count),
            actuator_embedding_dim=int(old.actuator_embedding_dim),
            delta_flow_scale=old.delta_flow_scale.detach().clone(),
        )
        self.v128_contract = ACTION_IDENTIFIABLE_MODEL_CONTRACT
        self.runtime_metadata.update(
            {
                "action_identifiable_model_contract": ACTION_IDENTIFIABLE_MODEL_CONTRACT,
                "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
                "edge_physics_required": True,
                "development_only": True,
                "full_promotion_allowed": False,
            }
        )


def build_action_identifiable_v128_model_from_graph(
    graph: Any,
    *,
    edge_artifact: EdgePhysicsArtifactV128,
    state_dim: int,
    rainfall_dim: int,
    delta_state_scale: torch.Tensor | np.ndarray,
    delta_flow_scale: torch.Tensor | np.ndarray,
    design: V128SurrogateDesign = V128SurrogateDesign(),
) -> ActionIdentifiableEdgeAwareSurrogateV128:
    edge_artifact.validate(graph)
    model_step_seconds = int(getattr(design, "model_step_seconds", -1))
    control_update_seconds = int(getattr(design, "control_update_seconds", -1))
    prediction_horizon_steps = int(getattr(design, "prediction_horizon_steps", -1))
    free_control_horizon_steps = int(getattr(design, "free_control_horizon_steps", -1))
    if (
        model_step_seconds,
        control_update_seconds,
        prediction_horizon_steps,
        free_control_horizon_steps,
    ) != (300, 600, 72, 24):
        raise ValueError("action-identifiable V128 received an incompatible time/horizon design")
    return ActionIdentifiableEdgeAwareSurrogateV128(
        edge_artifact=edge_artifact,
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


def _teacher_step(
    model: Any,
    *,
    chunk: dict[str, torch.Tensor],
    k: int,
    static: dict[str, torch.Tensor],
    physics_norm: torch.Tensor,
    identity: torch.Tensor | None,
    static_norm: torch.Tensor,
    edges: torch.Tensor,
    inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
    prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
    setting = chunk["settings"][:, k]
    q, response = model.actuator.forward_prepared(
        prev_state[:, static["up"]],
        prev_state[:, static["down"]],
        setting,
        prev_flow,
        physics_norm,
        identity,
    )
    injection = torch.zeros(
        prev_state.shape[0],
        prev_state.shape[1],
        1,
        device=prev_state.device,
        dtype=prev_state.dtype,
    )
    injection = injection.index_add(1, static["up"], -q[..., None]).index_add(
        1, static["down"], q[..., None]
    )
    action_context = model._typed_action_context(
        state=prev_state,
        setting=setting,
        previous_flow=prev_flow,
        predicted_flow=q,
        responsiveness=response,
        upstream=static["up"],
        downstream=static["down"],
        physics_norm=physics_norm,
        identity_embedding=identity,
    )
    pred_state = model.transition.forward_prepared(
        prev_state,
        chunk["rainfall"][:, k],
        static_norm,
        injection,
        edges,
        inv,
        action_context,
    )
    return pred_state, q


def train_action_identifiable_hydraulic_stage_v128(
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
) -> list[dict[str, float | int | str | bool]]:
    """Stage A absolute hydraulic fit plus direct candidate-reference effect supervision."""
    design.validate()
    torch.manual_seed(design.seed)
    np.random.seed(design.seed)
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    history: list[dict[str, float | int | str | bool]] = []
    phase_seen: dict[str, set[int]] = {
        name: set() for values in source_groups.values() for name in values
    }
    _reset_cuda_peak(device)

    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        group_losses: list[float] = []
        delta_state_reports: list[float] = []
        delta_flow_reports: list[float] = []
        transition_count = 0
        for source, name in _ordered(source_groups, epoch, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            steps = len(range(phase, horizon, design.teacher_stride))
            if steps <= 0:
                raise RuntimeError("action-identifiable Stage A selected no transitions")
            optimizer.zero_grad(set_to_none=True)
            absolute_report = 0.0
            phase_seen[name].add(phase)
            transition_count += int(steps * branches)

            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                chunk_n = len(positions)
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=chunk_n
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=chunk_n, dtype=torch.float32
                )
                chunk_loss = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    pred_state, q = _teacher_step(
                        model,
                        chunk=chunk,
                        k=k,
                        static=static,
                        physics_norm=physics_norm,
                        identity=identity,
                        static_norm=static_norm,
                        edges=edges,
                        inv=inv,
                    )
                    state_error = (pred_state - chunk["states"][:, k]) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    chunk_loss = chunk_loss + F.smooth_l1_loss(
                        state_error * state_weights,
                        torch.zeros_like(state_error),
                        beta=0.5,
                    )
                    chunk_loss = chunk_loss + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (chunk_loss / steps) * (float(chunk_n) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: action-identifiable absolute Stage A loss non-finite")
                weighted.backward()
                absolute_report += float(weighted.detach())
                del chunk, chunk_loss, weighted

            candidate_count = branches - 1
            pair_budget = min(4, candidate_count)
            delta_state_loss = torch.zeros((), device=device)
            delta_flow_loss = torch.zeros((), device=device)
            if pair_budget > 0:
                selected = _candidate_permutation(
                    candidate_count, group_name=name, epoch=epoch, seed=design.seed
                )[:pair_budget]
                positions = np.concatenate((np.asarray([0], dtype=np.int64), selected))
                pair = _select_to_device(cpu, positions, device=device, include_truth=True)
                pair_n = len(positions)
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=pair_n
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=pair_n, dtype=torch.float32
                )
                state_terms: list[torch.Tensor] = []
                flow_terms: list[torch.Tensor] = []
                for k in range(phase, horizon, design.teacher_stride):
                    pred_state, q = _teacher_step(
                        model,
                        chunk=pair,
                        k=k,
                        static=static,
                        physics_norm=physics_norm,
                        identity=identity,
                        static_norm=static_norm,
                        edges=edges,
                        inv=inv,
                    )
                    pred_state_delta = pred_state[1:] - pred_state[0:1]
                    true_state_delta = pair["states"][1:, k] - pair["states"][0:1, k]
                    pred_flow_delta = q[1:] - q[0:1]
                    true_flow_delta = pair["flows"][1:, k] - pair["flows"][0:1, k]
                    state_terms.append(
                        _response_weighted_effect_loss(
                            pred_state_delta * state_weights,
                            true_state_delta * state_weights,
                            model.transition.delta_state_scale,
                        )
                    )
                    flow_terms.append(
                        _response_weighted_effect_loss(
                            pred_flow_delta,
                            true_flow_delta,
                            model.actuator.delta_flow_scale,
                        )
                    )
                delta_state_loss = torch.stack(state_terms).mean()
                delta_flow_loss = torch.stack(flow_terms).mean()
                contrast = delta_state_loss + delta_flow_loss
                if not bool(torch.isfinite(contrast)):
                    raise RuntimeError(f"{name}: action-identifiable contrast loss non-finite")
                contrast.backward()
                del pair, state_terms, flow_terms, contrast

            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            total_report = absolute_report + float(delta_state_loss.detach()) + float(delta_flow_loss.detach())
            group_losses.append(total_report)
            delta_state_reports.append(float(delta_state_loss.detach()))
            delta_flow_reports.append(float(delta_flow_loss.detach()))
            del cpu, delta_state_loss, delta_flow_loss

        row: dict[str, float | int | str | bool] = {
            "stage": "v128_action_identifiable_counterfactual_teacher_forced",
            "contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
            "epoch": epoch,
            "teacher_phase": phase,
            "loss": float(np.mean(group_losses)),
            "delta_state_effect_loss": float(np.mean(delta_state_reports)),
            "delta_flow_effect_loss": float(np.mean(delta_flow_reports)),
            "teacher_transitions": int(transition_count),
            "counterfactual_pair_budget_per_group": 4,
            "fit_only_action_effect_supervision": True,
            "typed_action_context_used": True,
            "min_group_teacher_phase_coverage": min(
                (len(values) / design.teacher_stride for values in phase_seen.values()), default=0.0
            ),
            **_cuda_peak(device),
        }
        history.append(row)
        print(
            "[V128_ACTION_IDENTIFIABLE_STAGE_A] "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
        gc.collect()
    if any(len(values) < design.teacher_stride for values in phase_seen.values()):
        raise RuntimeError("action-identifiable Stage A missed a teacher phase")
    return history


def train_action_identifiable_rollout_stage_v128(
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
) -> list[dict[str, float | int | str | bool]]:
    """B0 rollout fit that explicitly preserves candidate-reference hydraulic effects."""
    design.validate()
    static = _static(graph, device)
    state_weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.rollout_learning_rate, weight_decay=design.weight_decay
    )
    model.train().to(device)
    history: list[dict[str, float | int | str | bool]] = []
    _reset_cuda_peak(device)

    for epoch, horizon in enumerate(design.rollout_horizons, start=1):
        losses: list[float] = []
        delta_flow_reports: list[float] = []
        delta_state_reports: list[float] = []
        for source, name in _ordered(source_groups, epoch + 101, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches = int(cpu["settings"].shape[0])
            candidates = _candidate_permutation(
                branches - 1, group_name=name, epoch=epoch, seed=design.seed
            )[: min(design.rollout_candidates_per_group, branches - 1)]
            positions = np.concatenate((np.asarray([0], dtype=np.int64), candidates))
            chunk = _select_to_device(cpu, positions, device=device, horizon=horizon, include_truth=True)
            optimizer.zero_grad(set_to_none=True)
            output = model.rollout(
                chunk["initial"],
                chunk["rainfall"],
                chunk["settings"],
                chunk["previous_flow"],
                static["up"],
                static["down"],
                static["physics"],
                static["static"],
                static["edges"],
            )
            state_error = (output.states - chunk["states"]) / model.transition.state_std
            flow_error = (output.actuator_flows - chunk["flows"]) / model.actuator.flow_std
            absolute = F.smooth_l1_loss(
                state_error * state_weights,
                torch.zeros_like(state_error),
                beta=0.5,
            ) + design.flow_weight * F.smooth_l1_loss(
                flow_error, torch.zeros_like(flow_error), beta=0.5
            )
            pred_state_delta = output.states[1:] - output.states[0:1]
            true_state_delta = chunk["states"][1:] - chunk["states"][0:1]
            pred_flow_delta = output.actuator_flows[1:] - output.actuator_flows[0:1]
            true_flow_delta = chunk["flows"][1:] - chunk["flows"][0:1]
            delta_state = _response_weighted_effect_loss(
                pred_state_delta * state_weights,
                true_state_delta * state_weights,
                model.transition.delta_state_scale,
            )
            delta_flow = _response_weighted_effect_loss(
                pred_flow_delta,
                true_flow_delta,
                model.actuator.delta_flow_scale,
            )
            loss = absolute + delta_state + delta_flow
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: action-identifiable rollout loss non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            delta_state_reports.append(float(delta_state.detach()))
            delta_flow_reports.append(float(delta_flow.detach()))
            del cpu, chunk, output, loss, absolute, delta_state, delta_flow

        row: dict[str, float | int | str | bool] = {
            "stage": "v128_action_identifiable_autoregressive_rollout",
            "contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
            "epoch": epoch,
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon * 5),
            "candidates_per_group": int(design.rollout_candidates_per_group),
            "loss": float(np.mean(losses)),
            "delta_state_effect_loss": float(np.mean(delta_state_reports)),
            "delta_flow_effect_loss": float(np.mean(delta_flow_reports)),
            "fit_only_action_effect_supervision": True,
            **_cuda_peak(device),
        }
        history.append(row)
        print(
            "[V128_ACTION_IDENTIFIABLE_B0] "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
        gc.collect()
    return history


def _post_objective_action_anchor_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, float | int | str | bool]:
    """One cheap FIT-only H360 effect rehearsal after exact TFV pairwise training."""
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(design.objective_learning_rate) * 0.25, 1.0e-6),
        weight_decay=design.weight_decay,
    )
    losses: list[float] = []
    selected_pairs = 0
    _reset_cuda_peak(device)
    for source, name in _ordered(source_groups, 911, design.seed):
        cpu = _cpu_group(source_caches[source], name, normalization)
        candidate_count = int(cpu["settings"].shape[0]) - 1
        if candidate_count <= 0:
            del cpu
            continue
        candidate = _candidate_permutation(
            candidate_count, group_name=name, epoch=911, seed=design.seed
        )[:1]
        positions = np.concatenate((np.asarray([0], dtype=np.int64), candidate))
        chunk = _select_to_device(cpu, positions, device=device, horizon=72, include_truth=True)
        optimizer.zero_grad(set_to_none=True)
        output = model.rollout(
            chunk["initial"],
            chunk["rainfall"],
            chunk["settings"],
            chunk["previous_flow"],
            static["up"],
            static["down"],
            static["physics"],
            static["static"],
            static["edges"],
        )
        delta_state = _response_weighted_effect_loss(
            output.states[1:] - output.states[0:1],
            chunk["states"][1:] - chunk["states"][0:1],
            model.transition.delta_state_scale,
        )
        delta_flow = _response_weighted_effect_loss(
            output.actuator_flows[1:] - output.actuator_flows[0:1],
            chunk["flows"][1:] - chunk["flows"][0:1],
            model.actuator.delta_flow_scale,
        )
        loss = 0.5 * (delta_state + delta_flow)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{name}: post-objective action anchor non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        selected_pairs += 1
        del cpu, chunk, output, delta_state, delta_flow, loss
    row: dict[str, float | int | str | bool] = {
        "stage": "v128_post_exact_action_effect_anchor",
        "contract": POST_OBJECTIVE_ANCHOR_CONTRACT,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "selected_fit_pairs": int(selected_pairs),
        "horizon_steps": 72,
        "learning_rate_fraction_of_exact_objective": 0.25,
        "fit_only_action_effect_supervision": True,
        "changes_exact_pairwise_census": False,
        **_cuda_peak(device),
    }
    print(
        "[V128_ACTION_IDENTIFIABLE_POST_OBJECTIVE] "
        + " ".join(f"{key}={value}" for key, value in row.items()),
        flush=True,
    )
    gc.collect()
    return row


def train_action_identifiable_objective_stage_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    """Run the unchanged exact H360 pairwise objective then preserve hydraulic sensitivity."""
    history = list(
        train_objective_stage_streaming_v128(
            model,
            source_caches=source_caches,
            source_groups=source_groups,
            normalization=normalization,
            graph=graph,
            device=device,
            flood_rate_index=flood_rate_index,
            design=design,
        )
    )
    history.append(
        _post_objective_action_anchor_v128(
            model,
            source_caches=source_caches,
            source_groups=source_groups,
            normalization=normalization,
            graph=graph,
            device=device,
            design=design,
        )
    )
    return history


__all__ = [
    "ACTION_CONDITIONED_FLOW_SCALE_CONTRACT",
    "ACTION_IDENTIFIABLE_MODEL_CONTRACT",
    "ACTION_IDENTIFIABLE_TRAINING_CONTRACT",
    "ActionIdentifiableActuatorFlowModelV128",
    "ActionIdentifiableEdgeAwareSurrogateV128",
    "POST_OBJECTIVE_ANCHOR_CONTRACT",
    "build_action_identifiable_v128_model_from_graph",
    "derive_action_conditioned_residual_scales_v128",
    "train_action_identifiable_hydraulic_stage_v128",
    "train_action_identifiable_objective_stage_v128",
    "train_action_identifiable_rollout_stage_v128",
]
