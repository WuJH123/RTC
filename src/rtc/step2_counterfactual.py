from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .flood_volume import trapezoid_node_flood_volume


PAIRING_CONTRACT = "STEP2_COUNTERFACTUAL_PAIRING_V1"
LOSS_CONTRACT = "STEP2_COUNTERFACTUAL_ACTION_LOSS_V1"


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
    if name in ds.files:
        return ds[name].astype(str)
    return np.asarray([default] * int(ds["initial_state"].shape[0]))


def counterfactual_groups(ds) -> dict[str, list[int]]:
    """Return same-prefix group row indices from one Step2 shard.

    The compiler is expected to preserve checkpoint groups within shards. Group identity
    includes source kind, rainfall/event identity and checkpoint. This function deliberately
    does not use target values to construct groups.
    """

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
    action_sha = _array(ds, "action_or_sequence_sha256")
    base_sha = _array(ds, "base_action_sha256")
    data_role = np.char.lower(_array(ds, "data_role"))
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
    alternatives = sorted(
        (i for i in indices if i != ref),
        key=lambda i: action_sha[i],
    )
    if not alternatives:
        return []
    take = min(int(budget), len(alternatives))
    offset = (int(epoch) * take) % len(alternatives)
    chosen = [
        alternatives[(offset + j) % len(alternatives)]
        for j in range(take)
    ]
    return [(ref, int(idx)) for idx in chosen]


def same_prefix_diagnostic(
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    if initial_state.shape[0] != 2:
        raise ValueError("counterfactual training batch must contain exactly one pair")
    for name, value in (
        ("initial_state", initial_state),
        ("rainfall", rainfall),
        ("previous_actuator_flow", previous_flow),
    ):
        if not torch.allclose(value[0], value[1], atol=atol, rtol=0.0):
            raise ValueError(f"counterfactual pair violates same-prefix {name}")


def _normalized_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    floor_scale: torch.Tensor,
) -> torch.Tensor:
    true_delta = target[1] - target[0]
    pred_delta = predicted[1] - predicted[0]
    reduce_dims = tuple(range(true_delta.dim() - 1))
    rms = true_delta.detach().square().mean(dim=reduce_dims).sqrt()
    scale = torch.maximum(rms, floor_scale)
    return F.smooth_l1_loss(
        (pred_delta - true_delta) / scale,
        torch.zeros_like(pred_delta),
    )


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
) -> PairMetrics:
    """Action-sensitive pair loss for one same-prefix reference/candidate pair."""

    if initial_state.shape[0] != 2:
        raise ValueError("counterfactual loss requires a two-branch pair")
    if rollout_states.shape != target_states.shape:
        raise ValueError("predicted/target state shapes differ")
    if rollout_flows.shape != target_flows.shape:
        raise ValueError("predicted/target flow shapes differ")

    zero = rollout_states.sum() * 0.0
    absolute_state = (
        ((rollout_states - target_states) / state_std) ** 2
    ).mean()
    absolute_flow = (
        ((rollout_flows - target_flows) / flow_std) ** 2
    ).mean()

    delta_flow = _normalized_delta_loss(
        rollout_flows,
        target_flows,
        floor_scale=(0.02 * flow_std).reshape(-1),
    )

    if flow_only:
        total = weights.absolute_flow * absolute_flow + weights.delta_flow * delta_flow
        nan = zero.detach()
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

    pred_node = trapezoid_node_flood_volume(
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
        if full_horizon and exact_node_flood_volume_m3 is not None
        else true_traj_node.detach()
    )
    pred_tfv = pred_node.sum(dim=-1)
    true_tfv = truth_node.sum(dim=-1)
    pred_delta_tfv = pred_tfv[1] - pred_tfv[0]
    true_delta_tfv = true_tfv[1] - true_tfv[0]
    delta_scale = true_delta_tfv.detach().abs().clamp_min(1.0)
    delta_tfv = F.smooth_l1_loss(
        (pred_delta_tfv - true_delta_tfv) / delta_scale,
        torch.zeros_like(pred_delta_tfv),
    )

    meaningful = true_delta_tfv.detach().abs() > 1e-6
    if bool(meaningful):
        sign = true_delta_tfv.detach().sign()
        ranking = F.softplus(-sign * pred_delta_tfv / delta_scale)
        sign_correct = (pred_delta_tfv.detach().sign() == sign).to(dtype=torch.float32)
    else:
        ranking = zero
        sign_correct = torch.ones((), device=zero.device, dtype=torch.float32)

    exact_flood = zero
    if full_horizon and exact_node_flood_volume_m3 is not None:
        exact = exact_node_flood_volume_m3.clamp_min(0.0)
        node_loss = torch.square(
            torch.log1p(pred_node) - torch.log1p(exact)
        ).mean()
        total_loss = torch.square(
            torch.log1p(pred_node.sum(dim=-1))
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
    sensitivity = pred_delta_tfv.detach().abs() / delta_scale
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
