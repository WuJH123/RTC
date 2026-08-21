"""Accuracy-first Step3 query-margin representation using frozen Step2 latent context.

V14 compressed a full hydraulic state, rainfall field and managed-flow state into only 11 global
summary numbers. That was sufficient for some candidate ranking signal but the HOLD margin collapsed
to execute-all. V15 keeps the rank/HOLD decomposition and reuses the already-trained Step2 latent
representation instead of retraining Step1/Step2 or generating more truth.

The deployed numeric margin remains the only ACTION/HOLD score. ``hold_logit`` is an auxiliary
training signal used to force the representation to learn the HOLD boundary; it is never a direct
online control rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .direct_tfv_policy_return import encode_policy_return_action_token
from .direct_tfv_policy_return_portfolio_admission import CURRENT_THREE_FAMILY_SOURCES


DIRECT_TFV_QUERY_MARGIN_V2_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_LATENT_CONTEXT_V2_THREE_FAMILY_82CONTROL_109REP"
)
DIRECT_TFV_QUERY_MARGIN_V2_FEATURE_CONTRACT = (
    "FROZEN_STEP2_STATE_RAIN_AND_CHANGED_FACILITY_LATENT_WITH_AUX_HOLD_V1"
)
DIRECT_TFV_QUERY_MARGIN_V2_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_LATENT_CHECKPOINT_V2"
)
QUERY_MARGIN_V2_HIDDEN_DIM = 64


@dataclass(frozen=True)
class QueryMarginV2Output:
    predicted_returns_m3: torch.Tensor
    query_best_margin_m3: torch.Tensor
    hold_logit: torch.Tensor
    rank_scores_normalized: torch.Tensor
    relative_rank_normalized: torch.Tensor


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _normalize(
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    flow: torch.Tensor,
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
        (rainfall - rain_mean) / rain_std,
        (flow - flow_mean) / flow_std,
    )


def _legacy_candidate_features(
    *,
    candidate_targets: torch.Tensor,
    active_target: torch.Tensor,
    base_step2_scores_m3: torch.Tensor,
    candidate_sources: Sequence[str],
    supervisory_mask: np.ndarray | torch.Tensor,
    target_scale_m3: float,
) -> torch.Tensor:
    target = candidate_targets
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=target.device).reshape(-1)
    if tuple(mask.shape) != (109,) or int(mask.sum()) != 82:
        raise ValueError("V15 query margin requires the frozen 82/109 supervisory mask")
    passive = ~mask
    if bool(torch.any(torch.abs(target[:, passive] - active_target[None, passive]) > 1.0e-7)):
        raise ValueError("V15 candidate changed a passive model channel")
    delta = target - active_target[None]
    active_delta = delta[:, mask]
    changed = torch.count_nonzero(torch.abs(active_delta) > 1.0e-7, dim=1).to(target.dtype)
    mean_abs = torch.mean(torch.abs(active_delta), dim=1)
    max_abs = torch.amax(torch.abs(active_delta), dim=1)
    rms = torch.sqrt(torch.mean(torch.square(active_delta), dim=1).clamp_min(0.0))
    base = base_step2_scores_m3.reshape(-1).to(target.dtype) / float(target_scale_m3)
    family = torch.zeros((target.shape[0], 3), dtype=target.dtype, device=target.device)
    allowed = tuple(CURRENT_THREE_FAMILY_SOURCES)
    for row, source in enumerate(candidate_sources):
        if str(source) not in allowed:
            raise ValueError(f"V15 received non-current candidate family: {source}")
        family[row, allowed.index(str(source))] = 1.0
    return torch.cat(
        (
            base[:, None],
            (changed / 82.0)[:, None],
            mean_abs[:, None],
            max_abs[:, None],
            rms[:, None],
            family,
            torch.ones((target.shape[0], 1), dtype=target.dtype, device=target.device),
        ),
        dim=1,
    )


def build_query_margin_v2_features(
    *,
    step2_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_targets: torch.Tensor,
    base_step2_scores_m3: torch.Tensor,
    candidate_sources: Sequence[str],
    supervisory_mask: np.ndarray | torch.Tensor,
    target_scale_m3: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rich causal features without updating the Step2 representation."""
    if current_state.ndim != 2:
        raise ValueError("V15 current_state must be [node,state_feature]")
    if rainfall_scenarios.ndim != 4:
        raise ValueError("V15 rainfall_scenarios must be [scenario,H,node,rain_feature]")
    if tuple(previous_actuator_flow.reshape(-1).shape) != (109,):
        raise ValueError("V15 previous_actuator_flow must contain 109 channels")
    if tuple(active_target.shape) != (109,):
        raise ValueError("V15 active_target must be [109]")
    if candidate_targets.ndim != 2 or tuple(candidate_targets.shape[1:]) != (109,):
        raise ValueError("V15 candidate targets must be [K,109]")
    if not 1 <= int(candidate_targets.shape[0]) <= 3:
        raise ValueError("V15 requires 1--3 current portfolio candidates")
    if len(candidate_sources) != int(candidate_targets.shape[0]):
        raise ValueError("V15 candidate source count is not aligned")
    if not math.isfinite(float(target_scale_m3)) or float(target_scale_m3) <= 0.0:
        raise ValueError("V15 target scale must be finite and positive")

    device = current_state.device
    dtype = current_state.dtype
    state_norm, rain_norm, flow_norm = _normalize(
        current_state,
        rainfall_scenarios,
        previous_actuator_flow.reshape(-1),
        normalization,
    )
    scenarios = int(rain_norm.shape[0])
    state_batch = state_norm[None].expand(scenarios, -1, -1)
    global_state = torch.cat((state_batch.mean(dim=1), state_batch.amax(dim=1)), dim=-1)
    state_context = step2_model.global_state_encoder(global_state)
    rainfall_context = step2_model.rainfall_encoder(step2_model._rainfall_summary(rain_norm))
    state_summary = state_context.mean(dim=0)
    rain_mean = rainfall_context.mean(dim=0)
    rain_std = rainfall_context.std(dim=0, unbiased=False)
    flow3 = torch.stack(
        (
            flow_norm.mean(),
            flow_norm.std(unbiased=False),
            torch.abs(flow_norm).amax(),
        )
    )
    query_context = torch.cat((state_summary, rain_mean, rain_std, flow3), dim=0)

    legacy = _legacy_candidate_features(
        candidate_targets=candidate_targets,
        active_target=active_target,
        base_step2_scores_m3=base_step2_scores_m3,
        candidate_sources=candidate_sources,
        supervisory_mask=supervisory_mask,
        target_scale_m3=target_scale_m3,
    )
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=device).reshape(-1)
    upstream = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    downstream = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=dtype, device=device)
    flow_batch = flow_norm[None].expand(scenarios, -1)
    common = step2_model._facility_context(
        current_state=state_batch,
        previous_actuator_flow=flow_batch,
        actuator_upstream=upstream,
        actuator_downstream=downstream,
        actuator_physics=physics,
        state_context=state_context,
        rainfall_context=rainfall_context,
    )

    latent_features: list[torch.Tensor] = []
    horizon = int(rain_norm.shape[1])
    for candidate in candidate_targets:
        active_batch = active_target[None].expand(scenarios, -1)
        candidate_batch = candidate[None].expand(scenarios, -1)
        reference, action = encode_policy_return_action_token(
            active_batch,
            candidate_batch,
            horizon_steps=horizon,
            first_action_steps=2,
        )
        reference_latent = step2_model._sequence_latent(common, reference)
        candidate_latent = step2_model._sequence_latent(common, action)
        delta_latent = candidate_latent - reference_latent
        changed = mask & (torch.abs(candidate - active_target) > 1.0e-7)
        if not bool(changed.any()):
            raise ValueError("V15 candidate has no changed supervisory facility")
        selected = delta_latent[:, changed, :]
        latent_mean = selected.mean(dim=(0, 1))
        latent_rms = torch.sqrt(torch.mean(torch.square(selected), dim=(0, 1)).clamp_min(0.0))
        latent_features.append(torch.cat((latent_mean, latent_rms), dim=0))
    candidate_features = torch.cat((legacy, torch.stack(latent_features)), dim=1)
    if not bool(torch.isfinite(query_context).all()) or not bool(torch.isfinite(candidate_features).all()):
        raise RuntimeError("V15 latent features contain non-finite values")
    return query_context, candidate_features


class QueryConditionedPolicyReturnAdapterV2(nn.Module):
    """Permutation-invariant rank/margin adapter with an auxiliary HOLD classifier."""

    def __init__(
        self,
        *,
        target_scale_m3: float,
        context_dim: int,
        candidate_dim: int,
        hidden_dim: int = QUERY_MARGIN_V2_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if min(int(context_dim), int(candidate_dim), int(hidden_dim)) <= 0:
            raise ValueError("V15 adapter dimensions must be positive")
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)
        self.register_buffer("target_scale_m3", torch.tensor(float(target_scale_m3), dtype=torch.float32))
        h = int(hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim + 1, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        self.rank_adjustment = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 1))
        joint_dim = 3 * h
        self.margin_head = nn.Sequential(
            nn.Linear(joint_dim, h), nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1)
        )
        self.hold_head = nn.Sequential(nn.Linear(joint_dim, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))

    def forward(
        self,
        *,
        raw_rank_scores_m3: torch.Tensor,
        context_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> QueryMarginV2Output:
        raw = raw_rank_scores_m3.reshape(-1)
        if tuple(context_features.shape) != (self.context_dim,):
            raise ValueError("V15 context feature width drifted")
        if candidate_features.ndim != 2 or tuple(candidate_features.shape) != (
            int(raw.numel()),
            self.candidate_dim,
        ):
            raise ValueError("V15 candidate feature shape drifted")
        scale = self.target_scale_m3.to(dtype=raw.dtype, device=raw.device).clamp_min(1.0)
        raw_norm = raw / scale
        context = self.context_encoder(context_features)
        candidates = self.candidate_encoder(torch.cat((candidate_features, raw_norm[:, None]), dim=1))
        rank = raw_norm + self.rank_adjustment(
            torch.cat((candidates, context[None].expand(candidates.shape[0], -1)), dim=1)
        ).squeeze(-1)
        relative = rank - torch.min(rank)
        pooled_mean = candidates.mean(dim=0)
        pooled_max = candidates.amax(dim=0)
        joint = torch.cat((context, pooled_mean, pooled_max), dim=0)
        margin = self.margin_head(joint).reshape(()) * scale
        hold_logit = self.hold_head(joint).reshape(())
        predicted = margin + relative * scale
        return QueryMarginV2Output(
            predicted_returns_m3=predicted,
            query_best_margin_m3=margin,
            hold_logit=hold_logit,
            rank_scores_normalized=rank,
            relative_rank_normalized=relative,
        )


__all__ = [
    "DIRECT_TFV_QUERY_MARGIN_V2_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_V2_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_V2_FEATURE_CONTRACT",
    "QUERY_MARGIN_V2_HIDDEN_DIM",
    "QueryConditionedPolicyReturnAdapterV2",
    "QueryMarginV2Output",
    "build_query_margin_v2_features",
]
