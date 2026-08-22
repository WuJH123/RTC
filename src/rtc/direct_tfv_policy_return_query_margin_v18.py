"""Step3 V18 compact selected-action/HOLD boundary on a frozen validated rank.

V17 showed that a high-capacity query-boundary network can collapse to one global logit even when
candidate ranking is already correct. V18 keeps the validated V15 rank unchanged and replaces the
deep boundary with a train-only standardized, PCA-compressed, L2-regularized linear score. The
ACTION/HOLD threshold is frozen from out-of-fold Train predictions rather than assuming score==0 is
an optimal decision threshold. No action quota is imposed: a collapsed held-out panel still fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


DIRECT_TFV_QUERY_MARGIN_V18_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_V18_FROZEN_V15_RANK_TRAIN_OOF_LINEAR_BOUNDARY"
)
DIRECT_TFV_QUERY_MARGIN_V18_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_QUERY_MARGIN_V18_CHECKPOINT_TRAIN_OOF_LINEAR_BOUNDARY"
)
PCA_COMPONENTS = 12
MAGNITUDE_COORDINATE_MAX = 6.0


@dataclass(frozen=True)
class BoundaryFeaturePartsV18:
    dense: torch.Tensor
    explicit: torch.Tensor


@dataclass(frozen=True)
class QueryMarginV18Output:
    hold_score: torch.Tensor
    decision_threshold: torch.Tensor
    boundary_distance: torch.Tensor
    query_best_margin_m3: torch.Tensor
    margin_coordinate: torch.Tensor
    magnitude_coordinate: torch.Tensor
    predicted_returns_m3: torch.Tensor
    selected_candidate_index: torch.Tensor
    relative_rank_normalized: torch.Tensor


def build_boundary_feature_parts_v18(
    *,
    context_features: torch.Tensor,
    candidate_features: torch.Tensor,
    raw_rank_scores_m3: torch.Tensor,
    rank_scores_normalized: torch.Tensor,
    selected_candidate_index: torch.Tensor,
    target_scale_m3: float,
) -> BoundaryFeaturePartsV18:
    """Build deterministic query-varying features after frozen-rank candidate selection."""
    if context_features.ndim != 1:
        raise ValueError("V18 context features must be one-dimensional")
    if candidate_features.ndim != 2:
        raise ValueError("V18 candidate features must be [K,F]")
    k = int(candidate_features.shape[0])
    if not 1 <= k <= 3:
        raise ValueError("V18 expects 1--3 portfolio candidates")
    raw = raw_rank_scores_m3.reshape(-1)
    rank = rank_scores_normalized.reshape(-1)
    if int(raw.numel()) != k or int(rank.numel()) != k:
        raise ValueError("V18 rank/candidate count mismatch")
    scale = float(target_scale_m3)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("V18 target scale must be finite and positive")
    selected = int(selected_candidate_index.detach().cpu())
    if selected < 0 or selected >= k:
        raise ValueError("V18 selected candidate index is out of range")

    selected_candidate = candidate_features[selected]
    pooled_mean = candidate_features.mean(dim=0)
    pooled_max = candidate_features.amax(dim=0)
    dense = torch.cat((context_features, selected_candidate, pooled_mean, pooled_max), dim=0)

    raw_norm = raw / scale
    sorted_rank = torch.sort(rank).values
    rank_gap = (
        sorted_rank[1] - sorted_rank[0]
        if int(sorted_rank.numel()) >= 2
        else rank.new_zeros(())
    )
    explicit = torch.stack(
        (
            raw_norm[selected],
            rank[selected],
            rank_gap,
            raw_norm.amax() - raw_norm.amin(),
            raw.new_tensor(float(k) / 3.0),
        )
    )
    if not bool(torch.isfinite(dense).all()) or not bool(torch.isfinite(explicit).all()):
        raise RuntimeError("V18 boundary features contain non-finite values")
    return BoundaryFeaturePartsV18(dense=dense, explicit=explicit)


class BoundaryPreprocessorV18(nn.Module):
    """Frozen Train-only standardization and PCA projection."""

    def __init__(
        self,
        *,
        dense_mean: torch.Tensor,
        dense_std: torch.Tensor,
        components: torch.Tensor,
        explicit_mean: torch.Tensor,
        explicit_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if dense_mean.ndim != 1 or dense_std.shape != dense_mean.shape:
            raise ValueError("V18 dense standardization shape mismatch")
        if components.ndim != 2 or int(components.shape[1]) != int(dense_mean.numel()):
            raise ValueError("V18 PCA component shape mismatch")
        if explicit_mean.ndim != 1 or explicit_std.shape != explicit_mean.shape:
            raise ValueError("V18 explicit standardization shape mismatch")
        self.register_buffer("dense_mean", dense_mean.detach().to(torch.float32))
        self.register_buffer("dense_std", dense_std.detach().to(torch.float32).clamp_min(1.0e-6))
        self.register_buffer("components", components.detach().to(torch.float32))
        self.register_buffer("explicit_mean", explicit_mean.detach().to(torch.float32))
        self.register_buffer("explicit_std", explicit_std.detach().to(torch.float32).clamp_min(1.0e-6))

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0] + self.explicit_mean.numel())

    def forward(self, parts: BoundaryFeaturePartsV18) -> torch.Tensor:
        dense = parts.dense.to(dtype=self.dense_mean.dtype, device=self.dense_mean.device)
        explicit = parts.explicit.to(dtype=self.explicit_mean.dtype, device=self.explicit_mean.device)
        if tuple(dense.shape) != tuple(self.dense_mean.shape):
            raise ValueError("V18 dense feature width drifted")
        if tuple(explicit.shape) != tuple(self.explicit_mean.shape):
            raise ValueError("V18 explicit feature width drifted")
        dense_z = (dense - self.dense_mean) / self.dense_std
        projected = self.components @ dense_z
        explicit_z = (explicit - self.explicit_mean) / self.explicit_std
        out = torch.cat((projected, explicit_z), dim=0)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("V18 transformed boundary feature is non-finite")
        return out


class LinearBoundaryCalibratorV18(nn.Module):
    """Frozen linear HOLD score, Train-OOF threshold, and signed numeric magnitude."""

    def __init__(
        self,
        *,
        preprocessor: BoundaryPreprocessorV18,
        boundary_weight: torch.Tensor,
        boundary_bias: float,
        decision_threshold: float,
        magnitude_weight: torch.Tensor,
        magnitude_bias: float,
        target_scale_m3: float,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        d = preprocessor.output_dim
        if tuple(boundary_weight.reshape(-1).shape) != (d,):
            raise ValueError("V18 boundary weight width mismatch")
        if tuple(magnitude_weight.reshape(-1).shape) != (d,):
            raise ValueError("V18 magnitude weight width mismatch")
        self.register_buffer("boundary_weight", boundary_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("boundary_bias", torch.tensor(float(boundary_bias), dtype=torch.float32))
        self.register_buffer("decision_threshold", torch.tensor(float(decision_threshold), dtype=torch.float32))
        self.register_buffer("magnitude_weight", magnitude_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("magnitude_bias", torch.tensor(float(magnitude_bias), dtype=torch.float32))
        self.register_buffer("target_scale_m3", torch.tensor(float(target_scale_m3), dtype=torch.float32))

    def score(self, parts: BoundaryFeaturePartsV18) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.preprocessor(parts)
        hold_score = torch.dot(self.boundary_weight.to(z.device), z) + self.boundary_bias.to(z.device)
        magnitude = torch.dot(self.magnitude_weight.to(z.device), z) + self.magnitude_bias.to(z.device)
        magnitude = torch.clamp(magnitude, min=1.0e-4, max=MAGNITUDE_COORDINATE_MAX)
        return hold_score, magnitude

    def predict(
        self,
        *,
        parts: BoundaryFeaturePartsV18,
        relative_rank_normalized: torch.Tensor,
        selected_candidate_index: torch.Tensor,
    ) -> QueryMarginV18Output:
        hold_score, magnitude = self.score(parts)
        threshold = self.decision_threshold.to(hold_score.device)
        boundary_distance = hold_score - threshold
        sign = torch.where(
            boundary_distance >= 0.0,
            hold_score.new_tensor(1.0),
            hold_score.new_tensor(-1.0),
        )
        coordinate = sign * magnitude
        scale = self.target_scale_m3.to(hold_score.device).clamp_min(1.0)
        margin = torch.sinh(coordinate) * scale
        relative = relative_rank_normalized.reshape(-1)
        predicted = margin + relative * scale
        return QueryMarginV18Output(
            hold_score=hold_score,
            decision_threshold=threshold,
            boundary_distance=boundary_distance,
            query_best_margin_m3=margin,
            margin_coordinate=coordinate,
            magnitude_coordinate=magnitude,
            predicted_returns_m3=predicted,
            selected_candidate_index=selected_candidate_index,
            relative_rank_normalized=relative,
        )


__all__ = [
    "BoundaryFeaturePartsV18",
    "BoundaryPreprocessorV18",
    "DIRECT_TFV_QUERY_MARGIN_V18_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_QUERY_MARGIN_V18_CONTRACT",
    "LinearBoundaryCalibratorV18",
    "MAGNITUDE_COORDINATE_MAX",
    "PCA_COMPONENTS",
    "QueryMarginV18Output",
    "build_boundary_feature_parts_v18",
]
