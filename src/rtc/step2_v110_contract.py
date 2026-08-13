"""Frozen scientific contract for Project7 Step2 V11.0.

V11 keeps the supported V7 direct signed Delta-TFV value model unchanged and
narrows the hydraulic task to the 0-120 minute decision-relevant response
window. It addresses lag, nonlocality, multi-actuator combination and rolling
10-minute MPC without requiring six-hour nodewise hydraulic precision.
"""
from __future__ import annotations
from dataclasses import dataclass

V110_CONTRACT = "PROJECT7_STEP2_V110_ACTUATOR_SET_DUALTIME_RESPONSE_V1"
V110_ACTIVE_EFFECT_CONTRACT = "PROJECT7_STEP2_V110_LOCAL_MEANINGFUL_EFFECT_V1"
V110_CURRENT_STEP2_CONTRACT = "PROJECT7_STEP2_CURRENT_V110_V1"

@dataclass(frozen=True)
class HydraulicHorizonV110:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    hydraulic_horizon_minutes: int = 120
    fine_until_minutes: int = 30
    fine_stride_steps: int = 1
    medium_stride_steps: int = 2

    @property
    def horizon_steps(self) -> int:
        return self.hydraulic_horizon_minutes * 60 // self.model_step_seconds

    @property
    def control_block_steps(self) -> int:
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("control update must be an integer multiple of model step")
        return self.control_update_seconds // self.model_step_seconds

    def validate(self) -> None:
        if self.model_step_seconds != 300:
            raise ValueError("V11 freezes the Project7 300-s model step")
        if self.control_update_seconds != 600:
            raise ValueError("V11 freezes the 600-s MPC update interval")
        if self.hydraulic_horizon_minutes != 120:
            raise ValueError("V11 hydraulic response horizon is frozen at 120 min")
        if self.fine_until_minutes != 30:
            raise ValueError("V11 fine horizon is frozen at 30 min")
        if self.fine_stride_steps != 1 or self.medium_stride_steps != 2:
            raise ValueError("V11 retained-time strides are frozen at 5/10 min")
        _ = self.control_block_steps

    def indices(self) -> tuple[int, ...]:
        self.validate()
        fine_end = self.fine_until_minutes * 60 // self.model_step_seconds
        indices = list(range(0, fine_end, self.fine_stride_steps))
        start_medium = fine_end + self.medium_stride_steps - 1
        indices.extend(range(start_medium, self.horizon_steps, self.medium_stride_steps))
        if self.horizon_steps - 1 not in indices:
            indices.append(self.horizon_steps - 1)
        return tuple(sorted(set(int(i) for i in indices)))

    def response_minutes(self) -> tuple[float, ...]:
        return tuple((i + 1) * self.model_step_seconds / 60.0 for i in self.indices())

@dataclass(frozen=True)
class ActiveEffectContractV110:
    quantile: float = 0.90
    quantile_fraction: float = 0.25
    depth_floor_m: float = 0.01
    depth_capacity_fraction: float = 0.01
    flood_floor_m3s: float = 1.0e-5
    volume_floor_m3: float = 1.0e-3
    storage_capacity_fraction: float = 0.005
    node_flow_floor_m3s: float = 1.0e-4
    actuator_flow_floor_m3s: float = 1.0e-4

    def validate(self) -> None:
        if self.quantile != 0.90 or self.quantile_fraction != 0.25:
            raise ValueError("V11 active-effect quantile is frozen at P90 x 0.25")
        if min(self.depth_floor_m, self.flood_floor_m3s, self.volume_floor_m3,
               self.node_flow_floor_m3s, self.actuator_flow_floor_m3s) <= 0:
            raise ValueError("V11 physical floors must be positive")
        if self.depth_capacity_fraction != 0.01 or self.storage_capacity_fraction != 0.005:
            raise ValueError("V11 relative physical floors are frozen")

@dataclass(frozen=True)
class ActuatorSetHydraulicContractV110:
    hidden_dim: int = 64
    attention_dim: int = 48
    attention_heads: int = 4
    actuator_identity_dim: int = 16
    time_feature_dim: int = 4
    pair_hidden_dim: int = 24
    active_focal_gamma: float = 1.5
    dense_delta_weight: float = 0.20
    active_weight: float = 1.00
    sign_weight: float = 0.75
    magnitude_weight: float = 1.00
    temporal_difference_weight: float = 0.30
    d2_epochs: int = 4
    d3_epochs: int = 10
    d3_weight: float = 0.75
    d2_anchor_weight: float = 0.25
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    candidate_chunk_size: int = 4
    seed: int = 42

    def validate(self) -> None:
        frozen = {
            "hidden_dim": (self.hidden_dim, 64),
            "attention_dim": (self.attention_dim, 48),
            "attention_heads": (self.attention_heads, 4),
            "actuator_identity_dim": (self.actuator_identity_dim, 16),
            "time_feature_dim": (self.time_feature_dim, 4),
            "pair_hidden_dim": (self.pair_hidden_dim, 24),
            "d2_epochs": (self.d2_epochs, 4),
            "d3_epochs": (self.d3_epochs, 10),
            "candidate_chunk_size": (self.candidate_chunk_size, 4),
            "seed": (self.seed, 42),
        }
        bad = [k for k, (v, e) in frozen.items() if v != e]
        if bad:
            raise ValueError(f"V11 frozen values changed: {bad}")
        if self.attention_dim % self.attention_heads:
            raise ValueError("attention_dim must divide attention heads")
        if abs(self.d3_weight + self.d2_anchor_weight - 1.0) > 1e-12:
            raise ValueError("V11 D3/D2 weights must sum to one")
        if self.d3_weight != 0.75 or self.d2_anchor_weight != 0.25:
            raise ValueError("V11 freezes D3/D2 at 0.75/0.25")

__all__ = [
    "ActiveEffectContractV110", "ActuatorSetHydraulicContractV110",
    "HydraulicHorizonV110", "V110_ACTIVE_EFFECT_CONTRACT",
    "V110_CONTRACT", "V110_CURRENT_STEP2_CONTRACT",
]
