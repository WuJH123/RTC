"""Development spatial action-effect audit for current V128 Step2.

The audit uses held-out D2 same-prefix counterfactual labels to answer a focused question:
does the surrogate preserve the sign and magnitude of an actuator's flooding-volume effect as
graph distance increases?  It does not train the model and does not consume Validation/Final.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .spatial_diagnostics_v128 import (
    DEFAULT_DISTANCE_BINS,
    action_effect_by_distance,
    actuator_node_hops,
    distance_bin_name,
)
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _static, _truth_node_volume
from .step2_train_v127_streaming import _cpu_group, _select_to_device

STEP2_SPATIAL_ACTION_EFFECT_CONTRACT = (
    "PROJECT7_V128_D2_ACTUATOR_NODE_DISTANCE_ACTION_EFFECT_AUDIT_V1"
)


def _changed_actuator_per_candidate(
    settings: torch.Tensor,
    *,
    tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Infer the single changed actuator in each D2 candidate sequence.

    D2 is contractually single-actuator.  A zero-change or multi-actuator branch is therefore
    a data-contract failure rather than something the spatial audit should silently ignore.
    """
    values = settings.detach().cpu().numpy().astype(np.float32, copy=False)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("D2 spatial audit expects settings [reference+candidate,H,A]")
    reference = values[0:1]
    changed = np.max(np.abs(values[1:] - reference), axis=1) > float(tolerance)
    result: list[int] = []
    for row, mask in enumerate(changed, start=1):
        indices = np.flatnonzero(mask)
        if indices.size != 1:
            raise ValueError(
                f"D2 candidate branch {row} changes {indices.size} actuators; "
                "spatial single-actuator audit requires exactly one"
            )
        result.append(int(indices[0]))
    return np.asarray(result, dtype=np.int64)


def _finite_mean(rows: Sequence[dict[str, float | int]], key: str) -> float | None:
    values = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def evaluate_d2_spatial_action_effect_v128(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    branch_chunk: int = 4,
    effect_floor_m3: float = 1.0,
) -> dict[str, Any]:
    if branch_chunk <= 0:
        raise ValueError("D2 spatial audit branch_chunk must be positive")
    selected = [str(name) for name in names]
    if not selected:
        raise ValueError("D2 spatial audit requires at least one held-out D2 group")
    if any(not name.startswith("D2::") for name in selected):
        raise ValueError("D2 spatial audit received a non-D2 group")

    node_count = len(graph.node_ids)
    distances = actuator_node_hops(
        np.asarray(graph.edge_index, dtype=np.int64),
        node_count,
        np.asarray(graph.actuator_upstream, dtype=np.int64),
        np.asarray(graph.actuator_downstream, dtype=np.int64),
    )
    static = _static(graph, device)
    model.eval().to(device)
    by_bin: dict[str, list[dict[str, float | int]]] = {
        distance_bin_name(lo, hi): [] for lo, hi in DEFAULT_DISTANCE_BINS
    }
    group_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for name in selected:
            cpu = _cpu_group(cache, name, normalization)
            branches = int(cpu["settings"].shape[0])
            if branches < 2:
                raise RuntimeError(f"{name}: D2 group has no candidate")
            actuator_indices = _changed_actuator_per_candidate(cpu["settings"])
            truth = _truth_node_volume(cache, name)
            if truth.shape != (branches, node_count):
                raise RuntimeError(
                    f"{name}: authoritative node flood volume shape {truth.shape} "
                    f"does not match {(branches, node_count)}"
                )

            ref = _select_to_device(cpu, [0], device=device, include_truth=False)
            ref_output = model.objective_rollout(
                initial_state=ref["initial"],
                rainfall=ref["rainfall"],
                settings=ref["settings"],
                previous_actuator_flow=ref["previous_flow"],
                actuator_upstream=static["up"],
                actuator_downstream=static["down"],
                actuator_physics=static["physics"],
                static_node_features=static["static"],
                edge_index=static["edges"],
                flood_rate_index=flood_rate_index,
                priority_indices=None,
                dt_seconds=300.0,
            )
            pred_ref = ref_output.node_flood_volume_m3[0].detach().cpu().numpy()
            candidate_predictions: list[np.ndarray] = []
            for start in range(1, branches, int(branch_chunk)):
                stop = min(start + int(branch_chunk), branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=False)
                output = model.objective_rollout(
                    initial_state=chunk["initial"],
                    rainfall=chunk["rainfall"],
                    settings=chunk["settings"],
                    previous_actuator_flow=chunk["previous_flow"],
                    actuator_upstream=static["up"],
                    actuator_downstream=static["down"],
                    actuator_physics=static["physics"],
                    static_node_features=static["static"],
                    edge_index=static["edges"],
                    flood_rate_index=flood_rate_index,
                    priority_indices=None,
                    dt_seconds=300.0,
                )
                candidate_predictions.append(
                    output.node_flood_volume_m3.detach().cpu().numpy().astype(np.float32)
                )
                del chunk, output
            pred_candidates = np.concatenate(candidate_predictions, axis=0)
            if pred_candidates.shape != truth[1:].shape:
                raise RuntimeError(f"{name}: spatial audit prediction/label branch order mismatch")

            metrics = action_effect_by_distance(
                truth[0],
                truth[1:],
                pred_ref,
                pred_candidates,
                actuator_indices,
                distances,
                effect_floor_m3=float(effect_floor_m3),
            )
            serializable: dict[str, dict[str, float | int | None]] = {}
            for key, row in metrics.items():
                clean: dict[str, float | int | None] = {}
                for metric, value in row.items():
                    if isinstance(value, (int, np.integer)):
                        clean[metric] = int(value)
                    else:
                        number = float(value)
                        clean[metric] = number if np.isfinite(number) else None
                serializable[key] = clean
                by_bin[key].append(row)
            group_rows.append(
                {
                    "group": name,
                    "candidate_count": int(branches - 1),
                    "changed_actuator_indices": actuator_indices.tolist(),
                    "metrics": serializable,
                }
            )
            del cpu, ref, ref_output

    summary: dict[str, dict[str, float | int | None]] = {}
    for key, rows in by_bin.items():
        summary[key] = {
            "groups": int(len(rows)),
            "aggregation": "equal_weight_per_D2_group",
            "effect_mae_m3": _finite_mean(rows, "effect_mae_m3"),
            "effect_relative_mae": _finite_mean(rows, "effect_relative_mae"),
            "effect_sign_accuracy": _finite_mean(rows, "effect_sign_accuracy"),
            "mean_informative_sign_total_per_group": _finite_mean(rows, "informative_sign_total"),
        }

    near = summary.get("1-3", {}).get("effect_mae_m3")
    far = summary.get("13+", {}).get("effect_mae_m3")
    degradation = (
        float(far) / float(near)
        if near is not None and far is not None and float(near) > 0.0
        else None
    )
    return {
        "contract": STEP2_SPATIAL_ACTION_EFFECT_CONTRACT,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "source": "held-out D2 same-prefix authoritative SWMM counterfactuals",
        "distance_definition": "undirected shortest hops from either actuator endpoint",
        "prediction_volume": "hard surrogate node flood volume, not smooth MPC proxy",
        "effect_floor_m3": float(effect_floor_m3),
        "group_count": int(len(group_rows)),
        "summary": summary,
        "far_13_plus_to_near_1_3_effect_mae_ratio": degradation,
        "groups": group_rows,
    }


__all__ = [
    "STEP2_SPATIAL_ACTION_EFFECT_CONTRACT",
    "_changed_actuator_per_candidate",
    "evaluate_d2_spatial_action_effect_v128",
]
