"""Frozen scientific contracts for Project7 Step2 V6.0.

V6 is intentionally isolated from the V4/V5 surrogate classes and training losses.
It defines a lower-dimensional engineering-feasible control manifold, separate value
and hydraulic surrogate objectives, and a multi-resolution hydraulic horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

V60_CONTRACT = "PROJECT7_STEP2_V60_CONTROL_LATENT_DUAL_SURROGATE_V1"
V60_CONTROL_MANIFOLD_CONTRACT = "PROJECT7_V60_HYDRAULIC_ZONE_TEMPORAL_BASIS_V1"
V60_VALUE_CONTRACT = "PROJECT7_V60_CONTROL_VALUE_SURROGATE_V1"
V60_HYDRAULIC_CONTRACT = "PROJECT7_V60_HYDRAULIC_RESPONSE_SURROGATE_V1"
V60_D3_DATA_CONTRACT = "PROJECT7_V60_TARGETED_D3_CANDIDATE_MANIFOLD_V1"


@dataclass(frozen=True)
class MultiResolutionHorizonV60:
    model_step_seconds: int = 300
    horizon_steps: int = 72
    control_update_seconds: int = 600
    fine_until_minutes: int = 30
    medium_until_minutes: int = 120
    fine_stride_steps: int = 1
    medium_stride_steps: int = 2
    coarse_stride_steps: int = 6

    @property
    def control_block_steps(self) -> int:
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("control update must be an integer multiple of model step")
        return self.control_update_seconds // self.model_step_seconds

    @property
    def control_blocks(self) -> int:
        if self.horizon_steps % self.control_block_steps:
            raise ValueError("horizon must divide into complete control blocks")
        return self.horizon_steps // self.control_block_steps

    def validate(self) -> None:
        if self.model_step_seconds <= 0 or self.horizon_steps <= 0:
            raise ValueError("invalid V6 horizon")
        if self.fine_until_minutes <= 0 or self.medium_until_minutes <= self.fine_until_minutes:
            raise ValueError("multi-resolution breakpoints must increase")
        total_minutes = self.horizon_steps * self.model_step_seconds / 60
        if self.medium_until_minutes >= total_minutes:
            raise ValueError("medium breakpoint must be inside the horizon")
        _ = self.control_blocks
        if min(self.fine_stride_steps, self.medium_stride_steps, self.coarse_stride_steps) <= 0:
            raise ValueError("multi-resolution strides must be positive")

    def indices(self) -> tuple[int, ...]:
        """0-based model-step indices retained for hydraulic supervision/decoding."""
        self.validate()
        step_minutes = self.model_step_seconds / 60.0
        fine_end = int(round(self.fine_until_minutes / step_minutes))
        medium_end = int(round(self.medium_until_minutes / step_minutes))
        indices: list[int] = []
        indices.extend(range(0, fine_end, self.fine_stride_steps))
        start_medium = fine_end + self.medium_stride_steps - 1
        indices.extend(range(start_medium, medium_end, self.medium_stride_steps))
        start_coarse = medium_end + self.coarse_stride_steps - 1
        indices.extend(range(start_coarse, self.horizon_steps, self.coarse_stride_steps))
        if self.horizon_steps - 1 not in indices:
            indices.append(self.horizon_steps - 1)
        return tuple(sorted(set(int(i) for i in indices)))

    def weights(self) -> np.ndarray:
        """Approximate time-duration weights for retained hydraulic points."""
        idx = np.asarray(self.indices(), dtype=np.int64)
        endpoints = np.concatenate([np.asarray([-1], dtype=np.int64), idx])
        delta_steps = np.diff(endpoints).astype(np.float32)
        return delta_steps / max(float(delta_steps.mean()), 1e-6)


@dataclass(frozen=True)
class ControlBasisContractV60:
    spatial_zone_count: int = 8
    temporal_basis_count: int = 6
    max_setting_delta_per_update: float = 0.5
    coefficient_limit: float = 1.0

    def validate(self, *, actuator_count: int, control_blocks: int) -> None:
        if actuator_count <= 0 or control_blocks <= 0:
            raise ValueError("actuator/control-block counts must be positive")
        if not 1 <= self.spatial_zone_count <= actuator_count:
            raise ValueError("invalid hydraulic-zone count")
        if not 2 <= self.temporal_basis_count <= control_blocks:
            raise ValueError("invalid temporal-basis count")
        if self.max_setting_delta_per_update <= 0:
            raise ValueError("setting-rate contract must be positive")
        if self.coefficient_limit <= 0:
            raise ValueError("coefficient limit must be positive")


@dataclass(frozen=True)
class ValueLossContractV60:
    exact_delta_tfv_weight: float = 1.0
    group_centered_weight: float = 0.5
    listwise_rank_weight: float = 1.5
    regret_weight: float = 0.5


@dataclass(frozen=True)
class HydraulicLossContractV60:
    depth_weight: float = 1.0
    flooding_weight: float = 2.0
    storage_weight: float = 1.0
    managed_flow_weight: float = 0.75
    flooding_onset_weight: float = 1.0
    wet_node_gain: float = 2.0
    near_surcharge_gain: float = 3.0
    storage_proximity_gain: float = 2.0
    near_surcharge_start_ratio: float = 0.80
    onset_epsilon_m3s: float = 1e-7

    def validate(self) -> None:
        values = (
            self.depth_weight,
            self.flooding_weight,
            self.storage_weight,
            self.managed_flow_weight,
            self.flooding_onset_weight,
            self.wet_node_gain,
            self.near_surcharge_gain,
            self.storage_proximity_gain,
        )
        if any(value < 0 for value in values):
            raise ValueError("hydraulic loss weights/gains must be non-negative")
        if not 0.0 < self.near_surcharge_start_ratio < 1.0:
            raise ValueError("near-surcharge start ratio must lie in (0,1)")
        if self.onset_epsilon_m3s < 0:
            raise ValueError("flood onset epsilon must be non-negative")


def require_feature(names: Sequence[str], name: str) -> int:
    try:
        return tuple(names).index(name)
    except ValueError as exc:
        raise ValueError(f"required V6 feature missing from graph contract: {name}") from exc


__all__ = [
    "ControlBasisContractV60",
    "HydraulicLossContractV60",
    "MultiResolutionHorizonV60",
    "V60_CONTRACT",
    "V60_CONTROL_MANIFOLD_CONTRACT",
    "V60_D3_DATA_CONTRACT",
    "V60_HYDRAULIC_CONTRACT",
    "V60_VALUE_CONTRACT",
    "ValueLossContractV60",
    "require_feature",
]
