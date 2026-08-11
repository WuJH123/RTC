"""Read-only diagnostics for the V4.1 D3 magnitude pathway.

The helpers in this module deliberately do not train a model or touch SWMM.  They
make the action-energy, fixed-memory, and within-group ranking calculations used by
the bounded V4.2 pathway audit explicit and testable.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import torch


def action_descriptors_v42(
    delta_u: np.ndarray,
    *,
    step_minutes: float = 5.0,
    control_block_size: int = 2,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Return absolute and temporal action descriptors for one candidate.

    V4.1 expands each 36 control block into two 5-minute model steps.  The
    per-model-step descriptors remain available, while ``changed_control_blocks``
    uses that two-step expansion explicitly.
    """

    values = np.asarray(delta_u, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("delta_u must have shape [model_step, actuator]")
    if control_block_size <= 0:
        raise ValueError("control_block_size must be positive")
    active = np.abs(values) > float(tolerance)
    active_per_step = active.sum(axis=1).astype(int)
    changed_steps = np.flatnonzero(active.any(axis=1))
    block_count = int(np.ceil(values.shape[0] / control_block_size))
    active_per_control_block: list[int] = []
    changed_control_blocks = 0
    for block in range(block_count):
        start = block * control_block_size
        stop = min(values.shape[0], start + control_block_size)
        block_active = active[start:stop].any(axis=0)
        active_per_control_block.append(int(block_active.sum()))
        changed_control_blocks += int(block_active.any())
    if changed_steps.size:
        first = int(changed_steps[0])
        last = int(changed_steps[-1])
        duration_minutes = float((last - first + 1) * step_minutes)
        first_action_minute = float(first * step_minutes)
        last_action_minute = float(last * step_minutes)
    else:
        first = last = None
        duration_minutes = 0.0
        first_action_minute = last_action_minute = None
    return {
        "action_l1": float(np.abs(values).sum()),
        "action_l2": float(np.linalg.norm(values)),
        "action_linf": float(np.abs(values).max(initial=0.0)),
        "signed_delta_sum": float(values.sum()),
        "sum_delta_squared": float(np.square(values).sum()),
        "action_energy_l1": float(np.abs(values).sum()),
        "squared_action_energy": float(np.square(values).sum()),
        "active_actuator_count_per_block": active_per_step.tolist(),
        "active_actuator_count_per_control_block": active_per_control_block,
        "mean_active_actuator_count": float(active_per_step.mean()) if active_per_step.size else 0.0,
        "max_active_actuator_count": int(active_per_step.max(initial=0)),
        "cumulative_changed_actuator_count": int(active.any(axis=0).sum()),
        "changed_control_blocks": int(changed_control_blocks),
        "action_duration_minutes": duration_minutes,
        "duration_since_first_action_minutes": duration_minutes,
        "first_action_model_step": first,
        "last_action_model_step": last,
        "first_action_minute": first_action_minute,
        "last_action_minute": last_action_minute,
        "model_step_count": int(values.shape[0]),
        "actuator_count": int(values.shape[1]),
    }


def causal_memory_trace_v42(value: torch.Tensor, *, rho: float = 0.65) -> torch.Tensor:
    """Apply the current bounded causal memory along the ``-3`` horizon axis."""

    if value.ndim < 3:
        raise ValueError("value must have a horizon at axis -3")
    if not 0.0 <= float(rho) < 1.0:
        raise ValueError("rho must be in [0, 1)")
    horizon = value.shape[-3]
    response: list[torch.Tensor] = []
    memory = torch.zeros_like(value[..., 0, :, :]) if value.ndim >= 4 else torch.zeros_like(value[..., 0, :])
    for step in range(horizon):
        memory = float(rho) * memory + value[..., step, :, :]
        response.append(memory)
    return torch.stack(response, dim=-3)


def magnitude_stratum_v42(abs_effect: float, *, q33: float, q67: float) -> str:
    """Classify an absolute response using fixed Train-only q33/q67 boundaries."""

    value = abs(float(abs_effect))
    if value < float(q33):
        return "small"
    if value < float(q67):
        return "medium"
    return "large"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def rank_correlation_v42(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Tie-aware Spearman correlation, returning NaN for constant inputs."""

    left = np.asarray(predicted, dtype=np.float64).reshape(-1)
    right = np.asarray(truth, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2:
        return float("nan")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.ptp(left_rank) == 0.0 or np.ptp(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pearson_correlation_v42(left: np.ndarray, right: np.ndarray) -> float:
    """Finite-only Pearson diagnostic, returning NaN for constant inputs."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size < 2 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pairwise_accuracy_v42(predicted: np.ndarray, truth: np.ndarray) -> float:
    """All-pairs order accuracy, excluding ties in the true effect."""

    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    if pred.size < 2:
        return float("nan")
    correct = total = 0
    for i in range(pred.size):
        for j in range(i + 1, pred.size):
            true_gap = target[i] - target[j]
            if not np.isfinite(true_gap) or abs(true_gap) <= 1e-12:
                continue
            predicted_gap = pred[i] - pred[j]
            if np.isfinite(predicted_gap):
                total += 1
                correct += int(np.sign(predicted_gap) == np.sign(true_gap))
    return float(correct / total) if total else float("nan")


def topology_propagation_status_v42(source_path: str | Path) -> dict[str, Any]:
    """Inspect whether ``forward_group`` references ``edge_index`` in its body."""

    path = Path(source_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forward = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward_group":
            forward = node
            break
    if forward is None:
        raise ValueError("forward_group not found")
    references = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Name) and node.id == "edge_index":
            references.append((node.lineno, node.col_offset))
        elif isinstance(node, ast.Attribute) and node.attr == "edge_index":
            references.append((node.lineno, node.col_offset))
    active = bool(references)
    return {
        "edge_index_referenced_in_forward_group": active,
        "references": [{"line": int(line), "column": int(column)} for line, column in references],
        "status": "TOPOLOGY_PROPAGATION_ACTIVE" if active else "TOPOLOGY_PROPAGATION_NOT_YET_ACTIVE",
        "source": str(path),
    }


__all__ = [
    "action_descriptors_v42",
    "causal_memory_trace_v42",
    "magnitude_stratum_v42",
    "pairwise_accuracy_v42",
    "pearson_correlation_v42",
    "rank_correlation_v42",
    "topology_propagation_status_v42",
]
