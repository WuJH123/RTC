"""Balanced direct signed-effect objective for the V11.3 mechanism gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v113 import HydraulicEffectOutputV113
from .step2_train_response_v60 import V60GroupBatch


@dataclass(frozen=True)
class V113EffectScales:
    state_scale: torch.Tensor
    flow_scale: torch.Tensor
    state_threshold: torch.Tensor
    flow_threshold: torch.Tensor


def _balanced(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].mean()


def v113_effect_loss(
    output: HydraulicEffectOutputV113,
    batch: V60GroupBatch,
    scales: V113EffectScales,
    *,
    storage_mask: torch.Tensor,
    inactive_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    idx = output.horizon_indices
    true_ref_state = batch.true_reference_states.index_select(1, idx)
    true_cand_state = batch.true_candidate_states.index_select(2, idx)
    true_ref_flow = batch.true_reference_flows.index_select(1, idx)
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
    true_state = true_cand_state - true_ref_state[:, None]
    true_flow = true_cand_flow - true_ref_flow[:, None]
    pred_state = output.raw_delta_states_physical[..., [0, 2, 3, 4, 5]]
    pred_flow = output.raw_delta_flows_physical
    state_scale = scales.state_scale.to(pred_state)
    flow_scale = scales.flow_scale.to(pred_flow)
    state_thr = scales.state_threshold.to(pred_state)
    flow_thr = scales.flow_threshold.to(pred_flow)
    truth_state5 = true_state[..., [0, 2, 3, 4, 5]]
    state_active = truth_state5.abs() >= state_thr[None, None, None]
    state_active[..., 2] &= storage_mask.to(state_active.device)[None, None, None]
    flow_active = true_flow.abs() >= flow_thr[None, None, None]
    dense_state = F.smooth_l1_loss(pred_state / state_scale, truth_state5 / state_scale, reduction="none")
    dense_flow = F.smooth_l1_loss(pred_flow / flow_scale, true_flow / flow_scale, reduction="none")
    active_state = _balanced(dense_state, state_active)
    active_flow = _balanced(dense_flow, flow_active)
    inactive_state = _balanced(dense_state, ~state_active)
    inactive_flow = _balanced(dense_flow, ~flow_active)
    # Signed direction is trained only where an effect is physically active.
    sign_state = _balanced(F.relu(-(pred_state * truth_state5)), state_active)
    sign_flow = _balanced(F.relu(-(pred_flow * true_flow)), flow_active)
    total = (
        0.25 * dense_state.mean() + 0.25 * dense_flow.mean()
        + 0.75 * active_state + 0.75 * active_flow
        + inactive_weight * 0.5 * (inactive_state + inactive_flow)
        + 0.10 * 0.5 * (sign_state + sign_flow)
    )
    stats = {
        "loss": float(total.detach()), "dense_state": float(dense_state.mean().detach()),
        "dense_flow": float(dense_flow.mean().detach()), "active_state": float(active_state.detach()),
        "active_flow": float(active_flow.detach()), "inactive_state": float(inactive_state.detach()),
        "inactive_flow": float(inactive_flow.detach()), "sign_state": float(sign_state.detach()),
        "sign_flow": float(sign_flow.detach()),
        "active_state_fraction": float(state_active.float().mean().detach()),
        "active_flow_fraction": float(flow_active.float().mean().detach()),
    }
    return total, stats


def derive_v113_scales(
    state_scale: np.ndarray | torch.Tensor,
    flow_scale: np.ndarray | torch.Tensor,
    state_threshold: np.ndarray | torch.Tensor,
    flow_threshold: np.ndarray | torch.Tensor,
    *, device: torch.device | str = "cpu",
) -> V113EffectScales:
    return V113EffectScales(
        torch.as_tensor(state_scale, dtype=torch.float32, device=device).clamp_min(1e-6),
        torch.as_tensor(flow_scale, dtype=torch.float32, device=device).clamp_min(1e-6),
        torch.as_tensor(state_threshold, dtype=torch.float32, device=device).clamp_min(1e-6),
        torch.as_tensor(flow_threshold, dtype=torch.float32, device=device).clamp_min(1e-6),
    )


__all__ = ["V113EffectScales", "derive_v113_scales", "v113_effect_loss"]
