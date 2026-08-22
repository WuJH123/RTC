"""Project7 Step3 V21 deployment-aligned selected-candidate vs HOLD boundary.

V20 proved that a facility-resolved all-record sign task does not transfer to the actual deployment
unit: the frozen V15 rank first selects one candidate, and only that candidate is compared with HOLD.
V21 therefore keeps the rich frozen V20 candidate representation, but constructs exactly one
query-level boundary sample after frozen-rank selection. Sibling candidates contribute only causal,
online-available portfolio context; sibling truth is never used as an additional boundary label.

The selected candidate feature is zero for an exact HOLD action. Portfolio contrast/rank context are
multiplied by selected action mass, so the complete V21 feature is also exactly zero for HOLD. The
preprocessor is scale-only plus non-centered SVD, and the classifier has no intercept; therefore the
physical zero boundary remains structural rather than learned or tuned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


DIRECT_TFV_SELECTED_BOUNDARY_V21_CONTRACT = (
    "PROJECT7_STEP3_V21_DEPLOYMENT_ALIGNED_SELECTED_QUERY_BOUNDARY"
)
DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT = (
    "PROJECT7_STEP3_V21_CHECKPOINT_SELECTED_QUERY_ZERO_BOUNDARY"
)
DIRECT_TFV_SELECTED_BOUNDARY_V21_FEATURE_CONTRACT = (
    "V20_FACILITY_FEATURE_PLUS_SELECTED_PORTFOLIO_CONTEXT_NO_SIBLING_TRUTH_V1"
)
BOUNDARY_ZERO = 0.0
SVD_COMPONENTS = 12
MAGNITUDE_COORDINATE_MAX = 6.0


@dataclass(frozen=True)
class SelectedBoundaryPartsV21:
    """One deployment-aligned selected-query boundary feature."""

    feature: torch.Tensor


@dataclass(frozen=True)
class SelectedBoundaryPredictionV21:
    """Selected candidate sign and sign-isolated numeric advantage."""

    hold_score: torch.Tensor
    magnitude_coordinate: torch.Tensor
    advantage_m3: torch.Tensor
    execute: torch.Tensor


def build_selected_portfolio_feature_v21(
    *,
    candidate_features: torch.Tensor,
    rank_scores: torch.Tensor,
    selected_index: int,
    selected_action_mass: torch.Tensor | float,
) -> SelectedBoundaryPartsV21:
    """Build one query feature after frozen-rank selection without using sibling truth.

    All portfolio-only blocks are gated by selected action mass. If the selected action is HOLD,
    candidate_features[selected] is required to be zero and every additional block also becomes zero.
    """
    if candidate_features.ndim != 2 or int(candidate_features.shape[0]) < 1:
        raise ValueError("V21 candidate_features must be [K,D] with K>=1")
    if rank_scores.ndim != 1 or int(rank_scores.numel()) != int(candidate_features.shape[0]):
        raise ValueError("V21 rank_scores must align with candidates")
    k = int(candidate_features.shape[0])
    selected = int(selected_index)
    if selected < 0 or selected >= k:
        raise ValueError("V21 selected_index is out of range")
    if not bool(torch.isfinite(candidate_features).all()) or not bool(torch.isfinite(rank_scores).all()):
        raise ValueError("V21 query inputs contain non-finite values")

    dtype = candidate_features.dtype
    device = candidate_features.device
    mass = torch.as_tensor(selected_action_mass, dtype=dtype, device=device).reshape(())
    if not bool(torch.isfinite(mass)) or float(mass.detach().cpu()) < 0.0:
        raise ValueError("V21 selected action mass must be finite and non-negative")

    selected_feature = candidate_features[selected]
    if float(mass.detach().cpu()) <= 1.0e-9:
        if bool(torch.any(torch.abs(selected_feature) > 1.0e-7)):
            raise ValueError("V21 HOLD selected candidate must have a zero candidate feature")
        width = 3 * int(selected_feature.numel()) + 3
        return SelectedBoundaryPartsV21(
            feature=torch.zeros(width, dtype=dtype, device=device)
        )

    portfolio_mean = candidate_features.mean(dim=0)
    if k == 1:
        sibling_mean = torch.zeros_like(selected_feature)
        rank_gap = rank_scores.new_zeros(())
    else:
        sibling_mask = torch.ones(k, dtype=torch.bool, device=device)
        sibling_mask[selected] = False
        sibling_mean = candidate_features[sibling_mask].mean(dim=0)
        other = rank_scores[sibling_mask]
        rank_gap = torch.min(other) - rank_scores[selected]
    rank_spread = torch.max(rank_scores) - torch.min(rank_scores)
    candidate_fraction = rank_scores.new_tensor(float(k) / 3.0)

    # The selected candidate feature is already action-anchored by V20. Portfolio context is gated
    # so it cannot create a pure query offset at zero action.
    feature = torch.cat(
        (
            selected_feature,
            mass * (selected_feature - sibling_mean),
            mass * portfolio_mean,
            mass * torch.stack((rank_gap, rank_spread, candidate_fraction)),
        ),
        dim=0,
    )
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("V21 selected portfolio feature contains non-finite values")
    return SelectedBoundaryPartsV21(feature=feature)


class SelectedBoundaryPreprocessorV21(nn.Module):
    """Train-only scale normalization plus non-centered SVD; zero remains exactly zero."""

    def __init__(self, *, feature_scale: torch.Tensor, components: torch.Tensor) -> None:
        super().__init__()
        if feature_scale.ndim != 1 or int(feature_scale.numel()) <= 0:
            raise ValueError("V21 feature_scale must be a non-empty vector")
        if components.ndim != 2 or int(components.shape[1]) != int(feature_scale.numel()):
            raise ValueError("V21 SVD component shape mismatch")
        if not 1 <= int(components.shape[0]) <= SVD_COMPONENTS:
            raise ValueError("V21 SVD component count is invalid")
        self.register_buffer(
            "feature_scale",
            feature_scale.detach().to(torch.float32).clamp_min(1.0e-6),
        )
        self.register_buffer("components", components.detach().to(torch.float32))

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def forward(self, parts: SelectedBoundaryPartsV21) -> torch.Tensor:
        feature = parts.feature.to(
            dtype=self.feature_scale.dtype,
            device=self.feature_scale.device,
        )
        if tuple(feature.shape) != tuple(self.feature_scale.shape):
            raise ValueError("V21 feature width drifted")
        z = feature / self.feature_scale
        out = self.components @ z
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("V21 transformed feature contains non-finite values")
        return out


class SelectedBoundaryCalibratorV21(nn.Module):
    """No-intercept selected-query sign model plus sign-isolated magnitude model."""

    def __init__(
        self,
        *,
        preprocessor: SelectedBoundaryPreprocessorV21,
        boundary_weight: torch.Tensor,
        magnitude_weight: torch.Tensor,
        target_scale_m3: float,
    ) -> None:
        super().__init__()
        d = preprocessor.output_dim
        if tuple(boundary_weight.reshape(-1).shape) != (d,):
            raise ValueError("V21 boundary weight width mismatch")
        if tuple(magnitude_weight.reshape(-1).shape) != (d,):
            raise ValueError("V21 magnitude weight width mismatch")
        scale = float(target_scale_m3)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("V21 target scale must be finite and positive")
        self.preprocessor = preprocessor
        self.register_buffer("boundary_weight", boundary_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("magnitude_weight", magnitude_weight.reshape(-1).detach().to(torch.float32))
        self.register_buffer("target_scale_m3", torch.tensor(scale, dtype=torch.float32))

    def predict(self, parts: SelectedBoundaryPartsV21) -> SelectedBoundaryPredictionV21:
        z = self.preprocessor(parts)
        score = torch.dot(self.boundary_weight.to(z.device), z)
        magnitude = torch.abs(torch.dot(self.magnitude_weight.to(z.device), z))
        magnitude = torch.clamp(magnitude, min=0.0, max=MAGNITUDE_COORDINATE_MAX)
        is_zero = torch.abs(score) <= 1.0e-12
        sign = torch.where(score < 0.0, score.new_tensor(-1.0), score.new_tensor(1.0))
        coordinate = sign * magnitude
        advantage = torch.sinh(coordinate) * self.target_scale_m3.to(score.device)
        advantage = torch.where(is_zero, advantage.new_zeros(()), advantage)
        return SelectedBoundaryPredictionV21(
            hold_score=score,
            magnitude_coordinate=magnitude,
            advantage_m3=advantage,
            execute=score < BOUNDARY_ZERO,
        )


__all__ = [
    "BOUNDARY_ZERO",
    "DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT",
    "DIRECT_TFV_SELECTED_BOUNDARY_V21_CONTRACT",
    "DIRECT_TFV_SELECTED_BOUNDARY_V21_FEATURE_CONTRACT",
    "MAGNITUDE_COORDINATE_MAX",
    "SVD_COMPONENTS",
    "SelectedBoundaryCalibratorV21",
    "SelectedBoundaryPartsV21",
    "SelectedBoundaryPredictionV21",
    "SelectedBoundaryPreprocessorV21",
    "build_selected_portfolio_feature_v21",
]
