"""Counterfactual-first model and causal response primitives for Project7 V128.

The current Development surrogate separates temporal managed-flow evolution, the direct
setting-conditioned response measured only at a candidate's first action divergence while the
reference/candidate still share the same authoritative hydraulic prefix, and later network
feedback which belongs to autoregressive hydraulic trajectory learning.

No SWMM action-gradient labels are used. Differentiability is retained so gradients can later
serve as an online solver signal once counterfactual trajectory/ranking fidelity is demonstrated.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .edge_physics_current_v128 import EdgePhysicsArtifactV128
from .step2_action_identifiable_v128 import (
    ActionIdentifiableActuatorFlowModelV128,
    ActionIdentifiableEdgeAwareSurrogateV128,
)
from .step2_differentiable_v128 import V128SurrogateDesign
from .step2_train_v127_streaming import derive_residual_scales_streaming_v127

COUNTERFACTUAL_FIRST_MODEL_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_EDGE_PHYSICS_SURROGATE_DEV_V3"
)
COUNTERFACTUAL_FIRST_TRAINING_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_TRAJECTORY_BEFORE_GRADIENT_DEV_V3"
)
DIRECT_ACTION_FLOW_SCALE_CONTRACT = (
    "PROJECT7_V128_FIT_ONLY_FIRST_DIVERGENCE_SAME_PREFIX_ACTION_FLOW_SCALE_V3"
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


def first_direct_response_spec_numpy(
    arrays: dict[str, Any],
    *,
    reference: int,
    candidate: int,
    require_single_actuator: bool = True,
) -> dict[str, Any] | None:
    """Return the causal local response at the first setting-divergence transition."""
    ref_setting = _finite_float32(arrays["settings"][reference], label="reference settings")
    cand_setting = _finite_float32(arrays["settings"][candidate], label="candidate settings")
    if ref_setting.shape != cand_setting.shape or ref_setting.ndim != 2:
        raise ValueError("counterfactual settings must be matching [H,A] arrays")
    delta = cand_setting - ref_setting
    steps = np.flatnonzero(np.any(np.abs(delta) > _SETTING_TOL, axis=1))
    if steps.size == 0:
        return None
    step = int(steps[0])
    changed = np.flatnonzero(np.abs(delta[step]) > _SETTING_TOL)
    if changed.size == 0 or (require_single_actuator and changed.size != 1):
        return None
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
    """Derive separate temporal and direct-action managed-flow scales from TrainFit only."""
    state_scale, temporal_flow_scale, telemetry = derive_residual_scales_streaming_v127(
        caches_and_names, sample_rows=sample_rows
    )
    temporal = _finite_float32(temporal_flow_scale, label="temporal flow scale").reshape(-1)
    samples: list[list[float]] = [[] for _ in range(int(temporal.size))]
    direct_pairs = zero_pairs = feedback_excluded = multi_excluded = prefix_excluded = 0
    for cache, names in caches_and_names:
        for name in names:
            entry = cache.entry(str(name))
            arrays = entry.arrays
            ref = int(entry.reference_index)
            horizon = int(np.asarray(arrays["settings"][ref]).shape[0])
            for raw in entry.indices:
                candidate = int(raw)
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
                    multi_excluded += 1
                    feedback_excluded += max(horizon - first, 0)
                    continue
                spec = first_direct_response_spec_numpy(
                    arrays, reference=ref, candidate=candidate, require_single_actuator=True
                )
                if spec is None:
                    prefix_excluded += 1
                    feedback_excluded += max(horizon - first, 0)
                    continue
                direct_pairs += 1
                feedback_excluded += max(horizon - int(spec["step"]) - 1, 0)
                effect = abs(float(spec["true_flow_delta"]))
                if effect > _DIRECT_EFFECT_FLOOR:
                    samples[int(spec["actuator_index"])].append(effect)
                else:
                    zero_pairs += 1

    action_scale = np.full(int(temporal.size), 1.0e-5, dtype=np.float32)
    action_counts = np.zeros(int(temporal.size), dtype=np.int64)
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
            "direct_response_zero_pairs": int(zero_pairs),
            "direct_response_actuators": int(np.sum(action_counts > 0)),
            "feedback_horizon_samples_excluded_from_direct_scale": int(feedback_excluded),
            "multi_actuator_first_divergence_excluded": int(multi_excluded),
            "prefix_mismatch_excluded": int(prefix_excluded),
            "same_prefix_state_atol": _PREFIX_STATE_ATOL,
            "same_prefix_flow_atol": _PREFIX_FLOW_ATOL,
            "holdout_used_for_scale": False,
            "scale_is_engineering_constraint": False,
        }
    )
    return state_scale, temporal.astype(np.float32), action_scale, telemetry


class CounterfactualFirstActuatorFlowModelV128(ActionIdentifiableActuatorFlowModelV128):
    """Use separate numerical scales for temporal baseline and direct setting response."""

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
        action_delta = torch.tanh(centered * linear + centered_quadratic * quadratic) * action_scale
        predicted_flow = previous_flow + baseline_delta + action_delta
        responsiveness = torch.sigmoid(self.diagnostic_response_logit(z)).squeeze(-1)
        return predicted_flow, responsiveness


class CounterfactualFirstEdgeAwareSurrogateV128(ActionIdentifiableEdgeAwareSurrogateV128):
    contract = COUNTERFACTUAL_FIRST_MODEL_CONTRACT

    def __init__(
        self, *, direct_action_flow_scale: Any, edge_artifact: EdgePhysicsArtifactV128, **kwargs: Any
    ) -> None:
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


def direct_effect_loss(
    predicted: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Magnitude plus soft signed-direction loss for an already actuator-balanced direct pair."""
    denom = scale.to(device=predicted.device, dtype=predicted.dtype).clamp_min(1.0e-5)
    p, t = predicted / denom, truth / denom
    magnitude = F.smooth_l1_loss(p, t, beta=0.5)
    informative = torch.abs(truth) > _DIRECT_EFFECT_FLOOR
    if bool(informative.any()):
        direction = F.softplus(-torch.sign(t[informative]) * p[informative]).mean()
    else:
        direction = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
    return magnitude + _DIRECTION_WEIGHT * direction, magnitude, direction


def oracle_transition_prediction(
    model: Any,
    *,
    prev_state: torch.Tensor,
    previous_flow: torch.Tensor,
    setting: torch.Tensor,
    oracle_flow: torch.Tensor,
    rainfall: torch.Tensor,
    static: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict one hydraulic step while replacing model q with authoritative managed flow."""
    batch = int(prev_state.shape[0])
    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=batch)
    with torch.no_grad():
        _, response = model.actuator.forward_prepared(
            prev_state[:, static["up"]], prev_state[:, static["down"]], setting,
            previous_flow, physics_norm, identity
        )
    injection = torch.zeros(
        batch, prev_state.shape[1], 1, device=prev_state.device, dtype=prev_state.dtype
    )
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


__all__ = [
    "COUNTERFACTUAL_FIRST_MODEL_CONTRACT",
    "COUNTERFACTUAL_FIRST_TRAINING_CONTRACT",
    "DIRECT_ACTION_FLOW_SCALE_CONTRACT",
    "CounterfactualFirstActuatorFlowModelV128",
    "CounterfactualFirstEdgeAwareSurrogateV128",
    "build_counterfactual_first_v128_model_from_graph",
    "derive_direct_response_scales_v128",
    "direct_effect_loss",
    "first_direct_response_spec_numpy",
    "oracle_transition_prediction",
]
