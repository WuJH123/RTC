"""Optimizer-consistent Direct-TFV receding MPC V6.

V6 keeps the V5 TFV-only objective and optimizer-aware one-sided admission, but closes a remaining
support-geometry gap. Per-facility q95 radii plus a q95 changed-facility ceiling do not guarantee
that the complete 12 x K H120 action sequence lies inside the D3 HOLD-reference distribution seen
by authoritative SWMM training branches.

The differentiable decoder therefore contracts every proposed sequence radially toward the current
HOLD target until first-block joint action mass, cumulative H120 action mass, and H120 temporal total
variation all lie within the selected D3 TrainFit support quantile. The same contracted sequence is
scored and executed, preserving score == execute. No TFV/PFV weight or performance-tuned threshold
is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v5 import DirectTFVMPCResultV5, DirectTFVRecedingMPCV5


DIRECT_TFV_STEP3_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6"


@dataclass(frozen=True)
class DirectTFVMPCResultV6(DirectTFVMPCResultV5):
    joint_sequence_support_quantile: str = "q95"
    joint_sequence_first_block_l1: float = 0.0
    joint_sequence_h120_l1: float = 0.0
    joint_sequence_h120_total_variation_l1: float = 0.0
    joint_sequence_support_max_ratio: float = 0.0
    joint_sequence_support_binding: bool = False


class DirectTFVRecedingMPCV6(DirectTFVRecedingMPCV5):
    """V5 admission plus a differentiable D3-HOLD joint-sequence trust region."""

    policy_mode = "direct_tfv_all109_receding_mpc_v6"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        admission_calibration: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
    ) -> None:
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            admission_calibration=admission_calibration,
            design=design,
        )
        validate_direct_tfv_sequence_support(
            sequence_support,
            actuator_ids=graph.actuator_ids,
        )
        self.sequence_support = dict(sequence_support)

    def _sequence_support_quantile(self) -> str:
        quantile = str(self.active_support_quantile_effective())
        if quantile not in {"q90", "q95", "q99"}:
            raise ValueError(f"unsupported Direct-TFV V6 sequence-support quantile: {quantile}")
        return quantile

    def _joint_sequence_geometry_torch(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if tuple(sequence.shape) != (self.design.prediction_horizon_steps, 109):
            raise ValueError("Direct-TFV V6 sequence must be [H72,109]")
        if tuple(active_target.shape) != (109,):
            raise ValueError("Direct-TFV V6 active target must contain 109 settings")
        block_steps = int(self.design.control_block_steps)
        blocks = sequence.reshape(-1, block_steps, 109).mean(dim=1)
        free = blocks[: int(self.design.free_control_blocks)]
        delta = free - active_target[None]
        previous = torch.cat((torch.zeros_like(delta[:1]), delta[:-1]), dim=0)
        return {
            "first_block_l1": torch.sum(torch.abs(delta[0])),
            "h120_l1": torch.sum(torch.abs(delta)),
            "h120_total_variation_l1": torch.sum(torch.abs(delta - previous)),
        }

    def _contract_to_joint_sequence_support(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> torch.Tensor:
        geometry = self._joint_sequence_geometry_torch(sequence, active_target)
        quantile = self._sequence_support_quantile()
        scales: list[torch.Tensor] = []
        for metric in SEQUENCE_SUPPORT_METRICS:
            mass = geometry[metric]
            limit = torch.as_tensor(
                sequence_support_limit(self.sequence_support, metric, quantile),
                dtype=sequence.dtype,
                device=sequence.device,
            )
            safe_scale = torch.where(
                mass > 1.0e-12,
                limit / mass.clamp_min(1.0e-12),
                torch.ones_like(mass),
            )
            scales.append(safe_scale)
        scale = torch.clamp(torch.min(torch.stack(scales)), min=0.0, max=1.0)
        hold = active_target[None].expand_as(sequence)
        return hold + scale * (sequence - hold)

    def _decode_active_fractions(
        self,
        fractions: torch.Tensor,
        *,
        active_indices: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        sequence = super()._decode_active_fractions(
            fractions,
            active_indices=active_indices,
            active_target=active_target,
        )
        return self._contract_to_joint_sequence_support(sequence, active_target)

    def joint_sequence_support_diagnostics(
        self, sequence: torch.Tensor, active_target: torch.Tensor
    ) -> dict[str, float | bool | str]:
        geometry = self._joint_sequence_geometry_torch(sequence, active_target)
        quantile = self._sequence_support_quantile()
        ratios: list[float] = []
        result: dict[str, float | bool | str] = {
            "quantile": quantile,
        }
        for metric in SEQUENCE_SUPPORT_METRICS:
            mass = float(geometry[metric].detach().cpu())
            limit = sequence_support_limit(self.sequence_support, metric, quantile)
            ratio = 0.0 if limit <= 0.0 else mass / limit
            result[metric] = mass
            result[f"{metric}_limit"] = float(limit)
            result[f"{metric}_ratio"] = float(ratio)
            ratios.append(float(ratio))
        maximum = max(ratios, default=0.0)
        result["max_ratio"] = float(maximum)
        result["binding"] = bool(maximum >= 0.999)
        return result

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV6:
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor):
            raise ValueError("Direct-TFV V6 requires active_target")
        result = super().optimize(**kwargs)
        candidate = result.optimized_candidate_settings
        if candidate is None:
            candidate = result.settings
        diagnostics = self.joint_sequence_support_diagnostics(candidate, active_target)
        values = dict(vars(result))
        values.update(
            {
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "joint_sequence_support_quantile": str(diagnostics["quantile"]),
                "joint_sequence_first_block_l1": float(diagnostics["first_block_l1"]),
                "joint_sequence_h120_l1": float(diagnostics["h120_l1"]),
                "joint_sequence_h120_total_variation_l1": float(
                    diagnostics["h120_total_variation_l1"]
                ),
                "joint_sequence_support_max_ratio": float(diagnostics["max_ratio"]),
                "joint_sequence_support_binding": bool(diagnostics["binding"]),
            }
        )
        return DirectTFVMPCResultV6(**values)


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCResultV6",
    "DirectTFVRecedingMPCV6",
]
