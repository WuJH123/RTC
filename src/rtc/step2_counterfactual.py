from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .flood_volume import (
    smooth_trapezoid_node_flood_volume,
    trapezoid_node_flood_volume,
)

PAIRING_CONTRACT = "STEP2_COUNTERFACTUAL_PAIRING_V2_VECTOR_BATCHED"
LOSS_CONTRACT = "STEP2_COUNTERFACTUAL_ACTION_LOSS_V4_SMOOTH_TFV_PROXY"


@dataclass(frozen=True)
class CounterfactualLossWeights:
    absolute_state: float = 0.5
    absolute_flow: float = 0.5
    delta_state: float = 1.0
    delta_flow: float = 1.0
    delta_tfv: float = 2.0
    ranking: float = 1.0
    exact_flood: float = 0.5
    physical: float = 0.1


@dataclass(frozen=True)
class PairMetrics:
    total: torch.Tensor
    absolute_state: torch.Tensor
    absolute_flow: torch.Tensor
    delta_state: torch.Tensor
    delta_flow: torch.Tensor
    delta_tfv: torch.Tensor
    ranking: torch.Tensor
    exact_flood: torch.Tensor
    physical: torch.Tensor
    true_delta_tfv_m3: torch.Tensor
    predicted_delta_tfv_m3: torch.Tensor
    sensitivity_ratio: torch.Tensor
    sign_correct: torch.Tensor


def _array(ds, name: str, default: str = "") -> np.ndarray:
    files = ds.files if hasattr(ds, "files") else ds.keys()
    if name in files:
        return ds[name].astype(str)
    return np.asarray([default] * int(ds["initial_state"].shape[0]))


def counterfactual_groups(ds) -> dict[str, list[int]]:
    """Return same-prefix group row indices from one Step2 shard."""

    count = int(ds["initial_state"].shape[0])
    event = _array(ds, "event_id")
    rain = _array(ds, "rainfall_group")
    checkpoint = _array(ds, "checkpoint_id")
    source = _array(ds, "source_kind", "D2")
    if any(len(x) != count for x in (event, rain, checkpoint, source)):
        raise ValueError("Step2 provenance arrays do not align with branch rows")
    groups: dict[str, list[int]] = {}
    for i in range(count):
        key = "::".join((source[i], rain[i], event[i], checkpoint[i]))
        groups.setdefault(key, []).append(i)
    return groups


def reference_index(ds, indices: list[int]) -> int:
    """Choose a deterministic in-group reference without using outcome labels."""

    if not indices:
        raise ValueError("counterfactual group is empty")
    source_values = {
        str(value).strip().upper()
        for value in _array(ds, "source_kind", "D2")[indices]
    }
    if len(source_values) != 1:
        raise ValueError(
            f"counterfactual group mixes source kinds: {sorted(source_values)}"
        )
    action_sha = _array(ds, "action_or_sequence_sha256")
    base_sha = _array(ds, "base_action_sha256")
    data_role = [str(value).strip().lower() for value in _array(ds, "data_role")]
    if next(iter(source_values)) == "D3":
        hold_indices = [
            int(idx)
            for idx in indices
            if data_role[idx] == "d3_hold_reference"
        ]
        if len(hold_indices) != 1:
            raise ValueError(
                "D3 counterfactual group requires exactly one D3_HOLD_REFERENCE; "
                f"found {len(hold_indices)}"
            )
        return hold_indices[0]
    for idx in indices:
        if base_sha[idx] and base_sha[idx] == action_sha[idx]:
            return int(idx)
    for idx in indices:
        if data_role[idx] in {"base", "reference", "hold", "center"}:
            return int(idx)
    return int(min(indices, key=lambda i: action_sha[i]))


def rotated_reference_pairs(
    ds,
    indices: list[int],
    *,
    epoch: int,
    budget: int,
) -> list[tuple[int, int]]:
    """Pair the deterministic reference with a rotating subset of alternatives."""

    if budget <= 0 or len(indices) < 2:
        return []
    ref = reference_index(ds, indices)
    action_sha = _array(ds, "action_or_sequence_sha256")
    alternatives = sorted((i for i in indices if i != ref), key=lambda i: action_sha[i])
    if not alternatives:
        return []
    take = min(int(budget), len(alternatives))
    offset = (int(epoch) * take) % len(alternatives)
    chosen = [alternatives[(offset + j) % len(alternatives)] for j in range(take)]
    return [(ref, int(idx)) for idx in chosen]


def _as_pairs(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] < 2 or value.shape[0] % 2:
        raise ValueError(
            "counterfactual branch batch must contain consecutive reference/candidate pairs"
        )
    return value.reshape(value.shape[0] // 2, 2, *value.shape[1:])


def same_prefix_diagnostic(
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """Fail closed if any vectorized reference/candidate pair has a different prefix."""

    for name, value in (
        ("initial_state", initial_state),
        ("rainfall", rainfall),
        ("previous_actuator_flow", previous_flow),
    ):
        paired = _as_pairs(value)
        if not torch.allclose(paired[:, 0], paired[:, 1], atol=atol, rtol=0.0):
            raise ValueError(f"counterfactual pair violates same-prefix {name}")


def _normalized_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    floor_scale: torch.Tensor,
) -> torch.Tensor:
    """Response-weighted candidate-minus-reference loss."""

    pred_pair = _as_pairs(predicted)
    true_pair = _as_pairs(target)
    true_delta = true_pair[:, 1] - true_pair[:, 0]
    pred_delta = pred_pair[:, 1] - pred_pair[:, 0]
    reduce_dims = tuple(range(1, true_delta.dim() - 1))
    rms = true_delta.detach().square().mean(dim=reduce_dims).sqrt()
    floor = floor_scale.reshape(1, -1).to(device=rms.device, dtype=rms.dtype)
    scale = torch.maximum(rms, floor)
    expand = [scale.shape[0]] + [1] * (true_delta.dim() - 2) + [scale.shape[-1]]
    scale = scale.reshape(*expand)
    normalized_error = (pred_delta - true_delta) / scale
    element_loss = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        reduction="none",
    )
    response_weight = 1.0 + true_delta.detach().abs() / scale
    normalize_dims = tuple(range(1, response_weight.dim()))
    response_weight = response_weight / response_weight.mean(
        dim=normalize_dims, keepdim=True
    ).clamp_min(1e-6)
    return (element_loss * response_weight).mean()


def counterfactual_action_loss(
    *,
    initial_state: torch.Tensor,
    rollout_states: torch.Tensor,
    rollout_flows: torch.Tensor,
    target_states: torch.Tensor,
    target_flows: torch.Tensor,
    exact_node_flood_volume_m3: torch.Tensor | None,
    dt_seconds: torch.Tensor,
    state_std: torch.Tensor,
    flow_std: torch.Tensor,
    full_horizon: bool,
    weights: CounterfactualLossWeights,
    flow_only: bool = False,
    use_authoritative_endpoint_truth: bool | None = None,
    use_exact_flood_loss: bool | None = None,
) -> PairMetrics:
    """Action-sensitive loss for vectorized same-prefix reference/candidate pairs.

    Delta-TFV and ranking use a smooth positive flooding-volume proxy for *predicted*
    flooding rates. The previous hard clamp created a zero-gradient dead zone whenever the
    recurrent surrogate predicted both candidate and reference rates slightly below zero.
    Authoritative/physical volume comparisons still use the hard non-negative operator.
    """

    _ = _as_pairs(initial_state)
    endpoint_truth_enabled = (
        bool(full_horizon)
        if use_authoritative_endpoint_truth is None
        else bool(use_authoritative_endpoint_truth)
    )
    exact_flood_enabled = (
        bool(full_horizon)
        if use_exact_flood_loss is None
        else bool(use_exact_flood_loss)
    )
    if rollout_states.shape != target_states.shape:
        raise ValueError("predicted/target state shapes differ")
    if rollout_flows.shape != target_flows.shape:
        raise ValueError("predicted/target flow shapes differ")

    zero = rollout_states.sum() * 0.0
    absolute_state = (((rollout_states - target_states) / state_std) ** 2).mean()
    absolute_flow = (((rollout_flows - target_flows) / flow_std) ** 2).mean()
    delta_flow = _normalized_delta_loss(
        rollout_flows,
        target_flows,
        floor_scale=(0.02 * flow_std).reshape(-1),
    )

    if flow_only:
        total = weights.absolute_flow * absolute_flow + weights.delta_flow * delta_flow
        nan = torch.full((), float("nan"), device=zero.device)
        return PairMetrics(
            total=total,
            absolute_state=zero,
            absolute_flow=absolute_flow,
            delta_state=zero,
            delta_flow=delta_flow,
            delta_tfv=zero,
            ranking=zero,
            exact_flood=zero,
            physical=zero,
            true_delta_tfv_m3=nan,
            predicted_delta_tfv_m3=nan,
            sensitivity_ratio=nan,
            sign_correct=nan,
        )

    delta_state = _normalized_delta_loss(
        rollout_states,
        target_states,
        floor_scale=(0.02 * state_std).reshape(-1),
    )

    # The scale is derived only from the Train normalization already supplied to Step2.
    # A small Softplus scale preserves useful gradient just below zero without changing
    # authoritative SWMM truth or the hard physical endpoint loss.
    flood_proxy_scale = (0.01 * state_std[2].detach()).clamp_min(1e-4)
    pred_proxy_node = smooth_trapezoid_node_flood_volume(
        initial_state,
        rollout_states,
        flood_rate_index=2,
        dt_seconds=dt_seconds,
        softplus_scale_m3s=flood_proxy_scale,
    )
    pred_physical_node = trapezoid_node_flood_volume(
        initial_state,
        rollout_states,
        flood_rate_index=2,
        dt_seconds=dt_seconds,
    )
    true_traj_node = trapezoid_node_flood_volume(
        initial_state,
        target_states,
        flood_rate_index=2,
        dt_seconds=dt_seconds,
    )
    truth_node = (
        exact_node_flood_volume_m3.clamp_min(0.0)
        if endpoint_truth_enabled and exact_node_flood_volume_m3 is not None
        else true_traj_node.detach()
    )
    pred_tfv_pair = _as_pairs(pred_proxy_node.sum(dim=-1))
    true_tfv_pair = _as_pairs(truth_node.sum(dim=-1))
    pred_delta_tfv = pred_tfv_pair[:, 1] - pred_tfv_pair[:, 0]
    true_delta_tfv = true_tfv_pair[:, 1] - true_tfv_pair[:, 0]
    delta_scale = true_delta_tfv.detach().abs().clamp_min(1.0)
    normalized_error = (pred_delta_tfv - true_delta_tfv) / delta_scale
    delta_tfv = F.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error)
    )

    # The loss already uses a 1 m3 normalization floor. Do not report/optimize the sign of
    # sub-1 m3 effects as if they were meaningful RTC decisions.
    meaningful = true_delta_tfv.detach().abs() >= 1.0
    if bool(meaningful.any()):
        sign = true_delta_tfv.detach().sign()
        per_pair_ranking = F.softplus(-sign * pred_delta_tfv / delta_scale)
        ranking = per_pair_ranking[meaningful].mean()
        sign_correct = (
            pred_delta_tfv.detach().sign()[meaningful] == sign[meaningful]
        ).to(dtype=torch.float32).mean()
    else:
        ranking = zero
        sign_correct = torch.ones((), device=zero.device, dtype=torch.float32)

    exact_flood = zero
    if exact_flood_enabled and exact_node_flood_volume_m3 is not None:
        exact = exact_node_flood_volume_m3.clamp_min(0.0)
        node_loss = torch.square(
            torch.log1p(pred_physical_node) - torch.log1p(exact)
        ).mean()
        total_loss = torch.square(
            torch.log1p(pred_physical_node.sum(dim=-1))
            - torch.log1p(exact.sum(dim=-1))
        ).mean()
        exact_flood = node_loss + total_loss

    depth = rollout_states[..., 0]
    flood = rollout_states[..., 2]
    volume = rollout_states[..., 3]
    physical = (
        torch.relu(-depth / state_std[0]).square().mean()
        + torch.relu(-flood / state_std[2]).square().mean()
        + torch.relu(-volume / state_std[3]).square().mean()
    )

    total = (
        weights.absolute_state * absolute_state
        + weights.absolute_flow * absolute_flow
        + weights.delta_state * delta_state
        + weights.delta_flow * delta_flow
        + weights.delta_tfv * delta_tfv
        + weights.ranking * ranking
        + weights.exact_flood * exact_flood
        + weights.physical * physical
    )
    sensitivity = (pred_delta_tfv.detach().abs() / delta_scale).mean()
    return PairMetrics(
        total=total,
        absolute_state=absolute_state,
        absolute_flow=absolute_flow,
        delta_state=delta_state,
        delta_flow=delta_flow,
        delta_tfv=delta_tfv,
        ranking=ranking,
        exact_flood=exact_flood,
        physical=physical,
        true_delta_tfv_m3=true_delta_tfv.detach(),
        predicted_delta_tfv_m3=pred_delta_tfv.detach(),
        sensitivity_ratio=sensitivity,
        sign_correct=sign_correct,
    )
