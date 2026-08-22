"""Facility-resolved zero-anchored Step3 boundary for Project7.

V19 showed that an all-record low-rank advantage regression still failed the Train-OOF sign gate even
though the frozen V15 rank remained perfect and held-out Validation scores were strongly ordered.
The remaining representation bottleneck is spatial: the V15--V19 query features globally pooled the
hydraulic state and changed-facility action latent, discarding which supervisory facilities changed
and the local upstream/downstream hydraulic context in which those changes were applied.

V20 keeps the validated V15 rank frozen and learns only candidate-vs-HOLD sign/magnitude from a
facility-resolved counterfactual signature. Every feature is multiplied by the candidate-minus-HOLD
action effect; therefore an exact HOLD action maps to the all-zero design. The boundary model has no
intercept and uses the physical zero threshold, so candidate==HOLD implies score==0 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .direct_tfv_policy_return import encode_policy_return_action_token
from .direct_tfv_policy_return_portfolio_admission import CURRENT_THREE_FAMILY_SOURCES


DIRECT_TFV_FACILITY_BOUNDARY_V20_CONTRACT = (
    "PROJECT7_STEP3_V20_FACILITY_RESOLVED_ZERO_ANCHORED_SIGN_BOUNDARY"
)
DIRECT_TFV_FACILITY_BOUNDARY_V20_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_V20_CHECKPOINT_FACILITY_RESOLVED_ZERO_ANCHORED_SIGN"
)
DIRECT_TFV_FACILITY_BOUNDARY_V20_FEATURE_CONTRACT = (
    "FROZEN_STEP2_FACILITY_MAIN_EFFECT_LOCAL_ENDPOINT_ACTION_WEIGHTED_V1"
)
BOUNDARY_ZERO = 0.0
MAGNITUDE_COORDINATE_MAX = 6.0


@dataclass(frozen=True)
class FacilityBoundaryPartsV20:
    """One zero-anchored facility-resolved candidate feature vector."""

    feature: torch.Tensor


@dataclass(frozen=True)
class FacilityBoundaryPredictionV20:
    """Selected candidate-vs-HOLD sign and signed numeric advantage."""

    hold_score: torch.Tensor
    magnitude_coordinate: torch.Tensor
    advantage_m3: torch.Tensor
    execute: torch.Tensor


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _normalize_inputs(
    *,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    normalization: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = current_state.dtype
    device = current_state.device
    state_mean = _tensor(normalization.state_mean, dtype=dtype, device=device)
    state_std = _tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1.0e-6)
    rain_mean = _tensor(normalization.rainfall_mean, dtype=dtype, device=device)
    rain_std = _tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1.0e-6)
    flow_mean = _tensor(normalization.flow_mean, dtype=dtype, device=device)
    flow_std = _tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1.0e-6)
    return (
        (current_state - state_mean) / state_std,
        (rainfall_scenarios - rain_mean) / rain_std,
        (previous_actuator_flow.reshape(-1) - flow_mean) / flow_std,
    )


def _weighted_pair(
    values: torch.Tensor,
    signed_weight: torch.Tensor,
    absolute_weight: torch.Tensor,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values[:, None]
    signed = torch.sum(values * signed_weight[:, None], dim=0)
    absolute = torch.sum(values * absolute_weight[:, None], dim=0)
    return torch.cat((signed, absolute), dim=0)


def build_facility_boundary_parts_v20(
    *,
    step2_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_target: torch.Tensor,
    candidate_source: str,
    supervisory_mask: np.ndarray | torch.Tensor,
    target_scale_m3: float,
) -> FacilityBoundaryPartsV20:
    """Build spatially aligned candidate-minus-HOLD features from frozen Step2 decomposition."""
    if current_state.ndim != 2:
        raise ValueError("V20 current_state must be [node,state_feature]")
    if rainfall_scenarios.ndim != 4:
        raise ValueError("V20 rainfall_scenarios must be [scenario,H,node,rain_feature]")
    if tuple(previous_actuator_flow.reshape(-1).shape) != (109,):
        raise ValueError("V20 previous_actuator_flow must contain 109 channels")
    if tuple(active_target.shape) != (109,) or tuple(candidate_target.shape) != (109,):
        raise ValueError("V20 active/candidate target must be [109]")
    if any(parameter.requires_grad for parameter in step2_model.parameters()):
        raise ValueError("V20 feature extraction requires frozen Step2")
    mask = torch.as_tensor(
        supervisory_mask,
        dtype=torch.bool,
        device=candidate_target.device,
    ).reshape(-1)
    if tuple(mask.shape) != (109,) or int(mask.sum()) != 82:
        raise ValueError("V20 requires the frozen 82/109 supervisory mask")
    passive = ~mask
    if bool(torch.any(torch.abs(candidate_target[passive] - active_target[passive]) > 1.0e-7)):
        raise ValueError("V20 candidate changed a passive/reference-only channel")
    allowed = tuple(CURRENT_THREE_FAMILY_SOURCES)
    if str(candidate_source) not in allowed:
        raise ValueError(f"V20 received non-current candidate family: {candidate_source}")
    scale = float(target_scale_m3)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("V20 target scale must be finite and positive")

    delta_all = candidate_target - active_target
    delta = delta_all[mask]
    abs_delta = torch.abs(delta)
    action_mass = torch.sum(abs_delta)
    dtype = current_state.dtype
    device = current_state.device
    if float(action_mass.detach().cpu()) <= 1.0e-9:
        # Width depends only on frozen input/model dimensions; construct once with zero-valued blocks.
        state_width = int(current_state.shape[-1])
        rain_width = int(rainfall_scenarios.shape[-1])
        embed_width = int(step2_model.actuator_embedding.embedding_dim)
        scalar_width = 17 + len(allowed)
        local_width = 4 * state_width + 4 * rain_width + 2 * embed_width
        return FacilityBoundaryPartsV20(
            feature=torch.zeros(scalar_width + local_width, dtype=dtype, device=device)
        )

    signed_weight = torch.zeros(109, dtype=dtype, device=device)
    absolute_weight = torch.zeros(109, dtype=dtype, device=device)
    signed_weight[mask] = delta / action_mass
    absolute_weight[mask] = abs_delta / action_mass

    state_norm, rain_norm, flow_norm = _normalize_inputs(
        current_state=current_state,
        rainfall_scenarios=rainfall_scenarios,
        previous_actuator_flow=previous_actuator_flow,
        normalization=normalization,
    )
    scenarios = int(rain_norm.shape[0])
    horizon = int(rain_norm.shape[1])
    state_batch = state_norm[None].expand(scenarios, -1, -1)
    flow_batch = flow_norm[None].expand(scenarios, -1)
    active_batch = active_target[None].expand(scenarios, -1)
    candidate_batch = candidate_target[None].expand(scenarios, -1)
    reference, encoded = encode_policy_return_action_token(
        active_batch,
        candidate_batch,
        horizon_steps=horizon,
        first_action_steps=2,
    )
    upstream = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    downstream = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=dtype, device=device)

    with torch.no_grad():
        output = step2_model(
            current_state=state_batch,
            rainfall=rain_norm,
            reference_settings=reference,
            candidate_settings=encoded,
            previous_actuator_flow=flow_batch,
            actuator_upstream=upstream,
            actuator_downstream=downstream,
            actuator_physics=physics,
        )
    facility_effect = output.facility_main_effect_m3.reshape(scenarios, 109).mean(dim=0) / scale
    interaction = output.interaction_residual_m3.reshape(scenarios).mean() / scale
    total = output.total_delta_tfv_m3.reshape(scenarios).mean() / scale
    activity = output.action_activity.reshape(scenarios, 109).mean(dim=0)

    local_up_state = state_norm[upstream]
    local_down_state = state_norm[downstream]
    rain_node = rain_norm.mean(dim=(0, 1))
    local_up_rain = rain_node[upstream]
    local_down_rain = rain_node[downstream]
    ids = torch.arange(109, device=device)
    embedding = step2_model.actuator_embedding(ids).to(dtype=dtype)

    changed = abs_delta > 1.0e-7
    changed_fraction = torch.count_nonzero(changed).to(dtype) / 82.0
    mean_abs = abs_delta.mean()
    max_abs = abs_delta.amax()
    rms = torch.sqrt(torch.mean(torch.square(delta)).clamp_min(0.0))
    signed_mean = delta.mean()
    positive_mean = torch.clamp(delta, min=0.0).mean()
    negative_mean = torch.clamp(-delta, min=0.0).mean()
    main_active = facility_effect[mask]
    main_positive = torch.clamp(main_active, min=0.0).sum()
    main_negative = torch.clamp(-main_active, min=0.0).sum()
    main_abs = torch.abs(main_active).sum()
    weighted_main_signed = torch.sum(facility_effect * signed_weight)
    weighted_main_abs = torch.sum(facility_effect * absolute_weight)
    weighted_flow_signed = torch.sum(flow_norm * signed_weight)
    weighted_flow_abs = torch.sum(flow_norm * absolute_weight)
    weighted_activity = torch.sum(activity * absolute_weight)
    family = torch.zeros(len(allowed), dtype=dtype, device=device)
    family[allowed.index(str(candidate_source))] = rms

    scalar = torch.cat(
        (
            torch.stack(
                (
                    total,
                    interaction,
                    facility_effect.sum(),
                    main_positive,
                    main_negative,
                    main_abs,
                    weighted_main_signed,
                    weighted_main_abs,
                    changed_fraction,
                    mean_abs,
                    max_abs,
                    rms,
                    signed_mean,
                    positive_mean,
                    negative_mean,
                    weighted_flow_signed,
                    weighted_flow_abs,
                    weighted_activity,
                )
            ),
            family,
        ),
        dim=0,
    )
    local = torch.cat(
        (
            _weighted_pair(local_up_state, signed_weight, absolute_weight),
            _weighted_pair(local_down_state, signed_weight, absolute_weight),
            _weighted_pair(local_up_rain, signed_weight, absolute_weight),
            _weighted_pair(local_down_rain, signed_weight, absolute_weight),
            _weighted_pair(embedding, signed_weight, absolute_weight),
        ),
        dim=0,
    )
    feature = torch.cat((scalar, local), dim=0)
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("V20 facility-resolved feature contains non-finite values")
    return FacilityBoundaryPartsV20(feature=feature)


class ScaleOnlyPreprocessorV20(nn.Module):
    """Train-only scale normalization that preserves the exact all-zero HOLD anchor."""

    def __init__(self, *, feature_scale: torch.Tensor) -> None:
        super().__init__()
        if feature_scale.ndim != 1 or int(feature_scale.numel()) <= 0:
            raise ValueError("V20 feature scale must be a non-empty vector")
        self.register_buffer(
            "feature_scale",
            feature_scale.detach().to(torch.float32).clamp_min(1.0e-6),
        )

    @property
    def output_dim(self) -> int:
        return int(self.feature_scale.numel())

    def forward(self, parts: FacilityBoundaryPartsV20) -> torch.Tensor:
        feature = parts.feature.to(
            dtype=self.feature_scale.dtype,
            device=self.feature_scale.device,
        )
        if tuple(feature.shape) != tuple(self.feature_scale.shape):
            raise ValueError("V20 feature width drifted")
        out = feature / self.feature_scale
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("V20 normalized feature contains non-finite values")
        return out


class FacilityBoundaryCalibratorV20(nn.Module):
    """No-intercept sign classifier plus sign-isolated magnitude model."""

    def __init__(
        self,
        *,
        preprocessor: ScaleOnlyPreprocessorV20,
        boundary_weight: torch.Tensor,
        magnitude_weight: torch.Tensor,
        target_scale_m3: float,
    ) -> None:
        super().__init__()
        d = preprocessor.output_dim
        if tuple(boundary_weight.reshape(-1).shape) != (d,):
            raise ValueError("V20 boundary weight width mismatch")
        if tuple(magnitude_weight.reshape(-1).shape) != (d,):
            raise ValueError("V20 magnitude weight width mismatch")
        scale = float(target_scale_m3)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("V20 target scale must be finite and positive")
        self.preprocessor = preprocessor
        self.register_buffer("boundary_weight", boundary_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("magnitude_weight", magnitude_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("target_scale_m3", torch.tensor(scale, dtype=torch.float32))

    def score(self, parts: FacilityBoundaryPartsV20) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.preprocessor(parts)
        hold_score = torch.dot(self.boundary_weight.to(z.device), z)
        magnitude = torch.abs(torch.dot(self.magnitude_weight.to(z.device), z))
        magnitude = torch.clamp(magnitude, min=0.0, max=MAGNITUDE_COORDINATE_MAX)
        return hold_score, magnitude

    def predict(self, parts: FacilityBoundaryPartsV20) -> FacilityBoundaryPredictionV20:
        score, magnitude = self.score(parts)
        is_zero = torch.abs(score) <= 1.0e-12
        sign = torch.where(score < 0.0, score.new_tensor(-1.0), score.new_tensor(1.0))
        coordinate = sign * magnitude
        advantage = torch.sinh(coordinate) * self.target_scale_m3.to(score.device)
        advantage = torch.where(is_zero, advantage.new_zeros(()), advantage)
        return FacilityBoundaryPredictionV20(
            hold_score=score,
            magnitude_coordinate=magnitude,
            advantage_m3=advantage,
            execute=score < BOUNDARY_ZERO,
        )


__all__ = [
    "BOUNDARY_ZERO",
    "DIRECT_TFV_FACILITY_BOUNDARY_V20_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_FACILITY_BOUNDARY_V20_CONTRACT",
    "DIRECT_TFV_FACILITY_BOUNDARY_V20_FEATURE_CONTRACT",
    "FacilityBoundaryCalibratorV20",
    "FacilityBoundaryPartsV20",
    "FacilityBoundaryPredictionV20",
    "MAGNITUDE_COORDINATE_MAX",
    "ScaleOnlyPreprocessorV20",
    "build_facility_boundary_parts_v20",
]
