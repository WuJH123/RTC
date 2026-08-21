"""Step3 V17: reuse a validated V15 rank branch and learn only selected-action vs HOLD.

The V16 experiment showed that reinitializing and retraining ranking can destroy a branch that was
already development-validated (V15: pairwise/top1 1.0/1.0 on the fixed Validation9). V17 therefore
supports importing the V15 rank subnetwork, freezes it, and trains a separate selected-candidate
boundary/magnitude model on existing exact-return truth only.

The deployed numeric margin has a sign that is *deterministically tied* to the HOLD boundary logit:

    hold_logit > 0  -> HOLD side  -> margin > 0
    hold_logit < 0  -> ACTION side -> margin < 0

Return magnitude is learned separately from |asinh(return / scale)| so large TFV magnitudes cannot
flip the boundary through a regression loss. Step2 remains frozen and no SWMM truth is generated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from .direct_tfv_policy_return import sha256_file
from .direct_tfv_policy_return_query_margin_v2 import QUERY_MARGIN_V2_HIDDEN_DIM


DIRECT_TFV_QUERY_MARGIN_V17_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_V17_FROZEN_V15_RANK_SELECTED_ACTION_BOUNDARY"
)
DIRECT_TFV_QUERY_MARGIN_V17_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_V17_CHECKPOINT_FROZEN_V15_RANK"
)
LEGACY_V15_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_LATENT_CHECKPOINT_V3_SELECTION_CONSISTENT"
)


@dataclass(frozen=True)
class QueryMarginV17Output:
    predicted_returns_m3: torch.Tensor
    query_best_margin_m3: torch.Tensor
    hold_logit: torch.Tensor
    margin_coordinate: torch.Tensor
    magnitude_coordinate: torch.Tensor
    rank_scores_normalized: torch.Tensor
    relative_rank_normalized: torch.Tensor
    selected_candidate_index: torch.Tensor


def _parameters(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    for module in modules:
        values.extend(module.parameters())
    return values


class QueryConditionedPolicyReturnAdapterV17(nn.Module):
    """Frozen imported rank plus a classifier-led numeric selected-action margin."""

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
            raise ValueError("V17 adapter dimensions must be positive")
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)
        self.register_buffer("target_scale_m3", torch.tensor(float(target_scale_m3), dtype=torch.float32))
        h = int(hidden_dim)

        # Exact structural copy of the V15 ranking path so validated V15 weights can be migrated.
        self.rank_context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        self.rank_candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim + 1, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        self.rank_adjustment = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 1))

        # Boundary and magnitude are separate tasks. Boundary alone determines sign.
        self.margin_context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        self.margin_candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim + 1, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()
        )
        joint_dim = 4 * h
        self.boundary_head = nn.Sequential(
            nn.Linear(joint_dim, h), nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1)
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(joint_dim, h), nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1)
        )

    def rank_parameters(self) -> list[nn.Parameter]:
        return _parameters((self.rank_context_encoder, self.rank_candidate_encoder, self.rank_adjustment))

    def margin_parameters(self) -> list[nn.Parameter]:
        return _parameters(
            (
                self.margin_context_encoder,
                self.margin_candidate_encoder,
                self.boundary_head,
                self.magnitude_head,
            )
        )

    def freeze_rank(self) -> None:
        for parameter in self.rank_parameters():
            parameter.requires_grad_(False)

    def train_margin_only(self) -> None:
        self.freeze_rank()
        for parameter in self.margin_parameters():
            parameter.requires_grad_(True)

    def forward(
        self,
        *,
        raw_rank_scores_m3: torch.Tensor,
        context_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> QueryMarginV17Output:
        raw = raw_rank_scores_m3.reshape(-1)
        if tuple(context_features.shape) != (self.context_dim,):
            raise ValueError("V17 context feature width drifted")
        if candidate_features.ndim != 2 or tuple(candidate_features.shape) != (
            int(raw.numel()),
            self.candidate_dim,
        ):
            raise ValueError("V17 candidate feature shape drifted")
        scale = self.target_scale_m3.to(dtype=raw.dtype, device=raw.device).clamp_min(1.0)
        raw_norm = raw / scale

        rank_context = self.rank_context_encoder(context_features)
        rank_candidates = self.rank_candidate_encoder(
            torch.cat((candidate_features, raw_norm[:, None]), dim=1)
        )
        rank = raw_norm + self.rank_adjustment(
            torch.cat((rank_candidates, rank_context[None].expand(rank_candidates.shape[0], -1)), dim=1)
        ).squeeze(-1)
        relative = rank - torch.min(rank)
        selected_index = torch.argmin(rank.detach())

        margin_context = self.margin_context_encoder(context_features)
        margin_candidates = self.margin_candidate_encoder(
            torch.cat((candidate_features, raw_norm[:, None]), dim=1)
        )
        selected_candidate = margin_candidates[selected_index]
        pooled_mean = margin_candidates.mean(dim=0)
        pooled_max = margin_candidates.amax(dim=0)
        joint = torch.cat((margin_context, selected_candidate, pooled_mean, pooled_max), dim=0)
        hold_logit = self.boundary_head(joint).reshape(())
        magnitude_coordinate = F.softplus(self.magnitude_head(joint).reshape(())) + 1.0e-4

        # Sign comes only from the boundary classifier; magnitude regression cannot flip it.
        margin_coordinate = torch.tanh(hold_logit) * magnitude_coordinate
        margin = torch.sinh(margin_coordinate) * scale
        predicted = margin + relative * scale
        return QueryMarginV17Output(
            predicted_returns_m3=predicted,
            query_best_margin_m3=margin,
            hold_logit=hold_logit,
            margin_coordinate=margin_coordinate,
            magnitude_coordinate=magnitude_coordinate,
            rank_scores_normalized=rank,
            relative_rank_normalized=relative,
            selected_candidate_index=selected_index,
        )


def import_v15_rank_state(
    adapter: QueryConditionedPolicyReturnAdapterV17,
    checkpoint: dict[str, Any],
) -> None:
    """Migrate only the V15 rank path; never import its failed HOLD-margin heads."""
    if str(checkpoint.get("contract", "")) != LEGACY_V15_CHECKPOINT_CONTRACT:
        raise ValueError("V17 rank source must be the V15 selection-consistent checkpoint")
    state = checkpoint.get("query_margin_state_dict")
    if not isinstance(state, dict):
        raise ValueError("V17 rank source lacks query_margin_state_dict")
    if int(checkpoint.get("context_dim", -1)) != adapter.context_dim:
        raise ValueError("V17/V15 context dimension mismatch")
    if int(checkpoint.get("candidate_dim", -1)) != adapter.candidate_dim:
        raise ValueError("V17/V15 candidate dimension mismatch")

    current = adapter.state_dict()
    mapping: dict[str, str] = {}
    for key in state:
        if key.startswith("context_encoder."):
            mapping[key] = "rank_context_encoder." + key[len("context_encoder.") :]
        elif key.startswith("candidate_encoder."):
            mapping[key] = "rank_candidate_encoder." + key[len("candidate_encoder.") :]
        elif key.startswith("rank_adjustment."):
            mapping[key] = key
    expected = {
        key
        for key in current
        if key.startswith("rank_context_encoder.")
        or key.startswith("rank_candidate_encoder.")
        or key.startswith("rank_adjustment.")
    }
    targets = set(mapping.values())
    if targets != expected:
        missing = sorted(expected - targets)
        extra = sorted(targets - expected)
        raise ValueError(f"V17 rank migration is incomplete: missing={missing}, extra={extra}")
    for source_key, target_key in mapping.items():
        value = state[source_key]
        if tuple(value.shape) != tuple(current[target_key].shape):
            raise ValueError(f"V17 rank tensor shape mismatch for {source_key}")
        current[target_key] = value.detach().clone()
    adapter.load_state_dict(current)
    adapter.freeze_rank()


def rank_state_sha256(adapter: QueryConditionedPolicyReturnAdapterV17) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(adapter.rank_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(index).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_query_margin_v17_checkpoint(
    checkpoint_path: str | Path,
    *,
    graph: Any,
    base_step2_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, QueryConditionedPolicyReturnAdapterV17, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("contract") != DIRECT_TFV_QUERY_MARGIN_V17_CHECKPOINT_CONTRACT:
        raise ValueError("runtime requires a V17 Step3 checkpoint")
    if str(payload.get("base_step2_sha256", "")).lower() != sha256_file(base_step2_path).lower():
        raise ValueError("V17 checkpoint/base Step2 SHA mismatch")
    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        base_step2_path,
        graph=graph,
        device=device,
    )
    for parameter in rank_model.parameters():
        parameter.requires_grad_(False)
    rank_model.eval()
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=float(rank_model.target_scale_m3.detach().cpu()),
        context_dim=int(payload["context_dim"]),
        candidate_dim=int(payload["candidate_dim"]),
    ).to(device)
    state = payload.get("query_margin_state_dict")
    if not isinstance(state, dict):
        raise ValueError("V17 checkpoint lacks adapter state")
    adapter.load_state_dict(state)
    adapter.freeze_rank()
    adapter.eval()
    return rank_model, normalization, adapter, payload


__all__ = [
    "DIRECT_TFV_QUERY_MARGIN_V17_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_V17_CONTRACT",
    "LEGACY_V15_CHECKPOINT_CONTRACT",
    "QueryConditionedPolicyReturnAdapterV17",
    "QueryMarginV17Output",
    "import_v15_rank_state",
    "load_query_margin_v17_checkpoint",
    "rank_state_sha256",
]
