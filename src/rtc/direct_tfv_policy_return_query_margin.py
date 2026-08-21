"""Query-conditioned decomposition for exact H10 receding-policy-return decisions.

The previous critic used one scalar for two different tasks: ranking candidates inside one same-prefix
query and deciding whether the best candidate beats HOLD=0.  A common scalar offset can therefore
improve candidate ranking while moving every candidate across zero.  This module removes that
non-identifiability from the deployed decision:

    rank_i       = learned candidate ordering score (lower is better)
    rel_i        = rank_i - min_j rank_j                         (best rel == 0)
    margin_q     = learned best-candidate return versus HOLD for the whole query
    return_hat_i = margin_q + target_scale * rel_i

Candidate selection depends only on relative rank. ACTION/HOLD depends only on the query-conditioned
best-candidate margin (plus the frozen one-sided conformal upper bound).  The adapter is intentionally
small and permutation invariant over the current 1--3 candidate portfolio.  Step1, the frozen base
Step2 checkpoint, the 109-channel representation, 82-control mask and candidate generator are not
changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
)
from .direct_tfv_policy_return_portfolio_admission import CURRENT_THREE_FAMILY_SOURCES


DIRECT_TFV_QUERY_MARGIN_CONTRACT = (
    "PROJECT7_EXACT_POLICY_RETURN_QUERY_CONDITIONED_MARGIN_V1_THREE_FAMILY"
)
DIRECT_TFV_QUERY_MARGIN_CHECKPOINT_CONTRACT = (
    "PROJECT7_EXACT_POLICY_RETURN_QUERY_MARGIN_CHECKPOINT_V1"
)
DIRECT_TFV_QUERY_MARGIN_FEATURE_CONTRACT = (
    "QUERY_CONTEXT11_CANDIDATE9_RELATIVE_RANK_WITH_SEPARATE_BEST_MARGIN_V1"
)
QUERY_MARGIN_CONTEXT_DIM = 11
QUERY_MARGIN_CANDIDATE_DIM = 9
QUERY_MARGIN_HIDDEN_DIM = 32


@dataclass(frozen=True)
class QueryMarginOutput:
    predicted_returns_m3: torch.Tensor
    query_best_margin_m3: torch.Tensor
    rank_scores_normalized: torch.Tensor
    relative_rank_normalized: torch.Tensor


def _finite_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")
    return value


def _summary4(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(-1)
    return torch.stack((flat.mean(), flat.std(unbiased=False), flat.amax(), flat.amin()))


def build_query_margin_features(
    *,
    current_state: torch.Tensor,
    rainfall_scenarios: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
    candidate_targets: torch.Tensor,
    base_step2_scores_m3: torch.Tensor,
    candidate_sources: Sequence[str],
    supervisory_mask: torch.Tensor | np.ndarray,
    target_scale_m3: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build causal fixed-width context/candidate features for one same-prefix query."""
    target = candidate_targets
    if target.ndim != 2 or int(target.shape[1]) != 109 or not 1 <= int(target.shape[0]) <= 3:
        raise ValueError("query-margin features require 1--3 candidate targets [K,109]")
    k = int(target.shape[0])
    if len(candidate_sources) != k:
        raise ValueError("candidate source count does not match query cardinality")
    if tuple(active_target.shape) != (109,):
        raise ValueError("active_target must be [109]")
    if tuple(previous_actuator_flow.shape) not in {(109,), (1, 109)}:
        raise ValueError("previous_actuator_flow must contain 109 channels")
    if int(base_step2_scores_m3.numel()) != k:
        raise ValueError("base Step2 score count does not match candidates")
    if not math.isfinite(float(target_scale_m3)) or float(target_scale_m3) <= 0.0:
        raise ValueError("target_scale_m3 must be finite and positive")
    mask = torch.as_tensor(supervisory_mask, dtype=torch.bool, device=target.device).reshape(-1)
    if tuple(mask.shape) != (109,) or int(mask.sum()) != 82:
        raise ValueError("query-margin features require the frozen 82/109 supervisory mask")
    passive = ~mask
    if bool(torch.any(torch.abs(target[:, passive] - active_target[None, passive]) > 1.0e-7)):
        raise ValueError("query-margin candidate changed a passive model channel")

    state4 = _summary4(current_state)
    rain4 = _summary4(rainfall_scenarios)
    flow = previous_actuator_flow.reshape(-1)
    flow3 = torch.stack((flow.mean(), flow.std(unbiased=False), torch.abs(flow).amax()))
    context = torch.cat((state4, rain4, flow3)).to(dtype=target.dtype, device=target.device)
    if tuple(context.shape) != (QUERY_MARGIN_CONTEXT_DIM,):
        raise RuntimeError("query-margin context feature dimension drifted")

    delta = target - active_target[None]
    changed = torch.count_nonzero(torch.abs(delta[:, mask]) > 1.0e-7, dim=1).to(target.dtype)
    active_delta = delta[:, mask]
    mean_abs = torch.mean(torch.abs(active_delta), dim=1)
    max_abs = torch.amax(torch.abs(active_delta), dim=1)
    rms = torch.sqrt(torch.mean(torch.square(active_delta), dim=1).clamp_min(0.0))
    base = base_step2_scores_m3.reshape(-1).to(dtype=target.dtype, device=target.device) / float(
        target_scale_m3
    )
    family = torch.zeros((k, 3), dtype=target.dtype, device=target.device)
    allowed = tuple(CURRENT_THREE_FAMILY_SOURCES)
    for row, source in enumerate(candidate_sources):
        if str(source) not in allowed:
            raise ValueError(f"query-margin received non-current candidate family: {source}")
        family[row, allowed.index(str(source))] = 1.0
    candidate = torch.cat(
        (
            base[:, None],
            (changed / 82.0)[:, None],
            mean_abs[:, None],
            max_abs[:, None],
            rms[:, None],
            family,
            torch.ones((k, 1), dtype=target.dtype, device=target.device),
        ),
        dim=1,
    )
    if tuple(candidate.shape) != (k, QUERY_MARGIN_CANDIDATE_DIM):
        raise RuntimeError("query-margin candidate feature dimension drifted")
    return _finite_tensor(context, label="query context"), _finite_tensor(
        candidate, label="query candidate features"
    )


class QueryConditionedPolicyReturnAdapter(nn.Module):
    """Small permutation-invariant adapter separating candidate rank from HOLD margin."""

    def __init__(self, *, target_scale_m3: float, hidden_dim: int = QUERY_MARGIN_HIDDEN_DIM) -> None:
        super().__init__()
        if not math.isfinite(float(target_scale_m3)) or float(target_scale_m3) <= 0.0:
            raise ValueError("query-margin adapter target scale must be finite and positive")
        if int(hidden_dim) <= 0:
            raise ValueError("query-margin hidden dimension must be positive")
        self.register_buffer("target_scale_m3", torch.tensor(float(target_scale_m3), dtype=torch.float32))
        h = int(hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(QUERY_MARGIN_CANDIDATE_DIM + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(QUERY_MARGIN_CONTEXT_DIM, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU()
        )
        self.rank_adjustment = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 1))
        self.margin_head = nn.Sequential(
            nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1)
        )

    def forward(
        self,
        *,
        raw_rank_scores_m3: torch.Tensor,
        context_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> QueryMarginOutput:
        raw = raw_rank_scores_m3.reshape(-1)
        k = int(raw.numel())
        if not 1 <= k <= 3 or tuple(candidate_features.shape) != (k, QUERY_MARGIN_CANDIDATE_DIM):
            raise ValueError("query-margin adapter requires 1--3 aligned candidate rows")
        if tuple(context_features.shape) != (QUERY_MARGIN_CONTEXT_DIM,):
            raise ValueError("query-margin adapter context has wrong width")
        scale = self.target_scale_m3.to(dtype=raw.dtype, device=raw.device).clamp_min(1.0)
        raw_norm = raw / scale
        candidate = self.candidate_encoder(torch.cat((candidate_features, raw_norm[:, None]), dim=1))
        context = self.context_encoder(context_features)
        ctx = context[None].expand(k, -1)
        rank = raw_norm + self.rank_adjustment(torch.cat((candidate, ctx), dim=1)).squeeze(-1)
        relative = rank - torch.min(rank)
        pooled_mean = candidate.mean(dim=0)
        pooled_max = candidate.amax(dim=0)
        margin_norm = self.margin_head(torch.cat((context, pooled_mean, pooled_max))).reshape(())
        margin = margin_norm * scale
        predicted = margin + relative * scale
        return QueryMarginOutput(
            predicted_returns_m3=_finite_tensor(predicted, label="query-margin predicted returns"),
            query_best_margin_m3=_finite_tensor(margin, label="query best margin"),
            rank_scores_normalized=_finite_tensor(rank, label="query rank scores"),
            relative_rank_normalized=_finite_tensor(relative, label="query relative ranks"),
        )


def load_query_margin_checkpoint(
    path: str | Path,
    *,
    graph: Any,
    base_step2_path: str | Path,
    device: torch.device,
) -> tuple[Any, Any, QueryConditionedPolicyReturnAdapter, dict[str, Any]]:
    """Load a frozen rank critic plus the query-conditioned margin adapter."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != DIRECT_TFV_QUERY_MARGIN_CHECKPOINT_CONTRACT:
        raise ValueError("runtime requires the current query-margin checkpoint")
    if payload.get("development_only") is not True:
        raise ValueError("query-margin checkpoint must be Development-only")
    if str(payload.get("estimand", "")) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("query-margin checkpoint has wrong policy-return estimand")
    if str(payload.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("query-margin checkpoint has wrong action encoding")
    import hashlib

    base_sha = hashlib.sha256(Path(base_step2_path).read_bytes()).hexdigest()
    if str(payload.get("base_step2_sha256", "")).lower() != base_sha.lower():
        raise ValueError("query-margin checkpoint was initialized from another Step2 checkpoint")
    if int(payload.get("supervisory_control_dimension", -1)) != 82 or int(
        payload.get("model_action_channel_count", -1)
    ) != 109:
        raise ValueError("query-margin checkpoint lost the frozen 82/109 contract")
    if int(payload.get("validation_rainfall_group_count", 0)) < 12:
        raise ValueError("query-margin checkpoint lacks fresh independent validation evidence")
    if payload.get("fresh_validation_verified") is not True:
        raise ValueError("query-margin checkpoint did not pass the fresh-validation firewall")
    rank_model, normalization, base = load_direct_tfv_runtime_checkpoint(
        base_step2_path, graph=graph, device=device
    )
    state_dict = payload.get("rank_model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("query-margin checkpoint lacks rank model state")
    rank_model.load_state_dict(state_dict, strict=True)
    rank_model.eval()
    adapter = QueryConditionedPolicyReturnAdapter(
        target_scale_m3=float(rank_model.target_scale_m3.detach().cpu()),
        hidden_dim=int(payload.get("query_margin_hidden_dim", QUERY_MARGIN_HIDDEN_DIM)),
    ).to(device)
    adapter_state = payload.get("query_margin_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("query-margin checkpoint lacks adapter state")
    adapter.load_state_dict(adapter_state, strict=True)
    adapter.eval()
    runtime = dict(payload)
    runtime.pop("rank_model_state_dict", None)
    runtime.pop("query_margin_state_dict", None)
    runtime["base_action_support"] = base["action_support"]
    return rank_model, normalization, adapter, runtime


__all__ = [
    "DIRECT_TFV_QUERY_MARGIN_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_FEATURE_CONTRACT",
    "QUERY_MARGIN_CANDIDATE_DIM",
    "QUERY_MARGIN_CONTEXT_DIM",
    "QueryConditionedPolicyReturnAdapter",
    "QueryMarginOutput",
    "build_query_margin_features",
    "load_query_margin_checkpoint",
]
