"""Development-only D2 TFV-gradient audit for nonfinal V128 stage checkpoints.

For each D2 group the exact reference H360 action sequence is differentiated once to obtain
full dJ/dU[72,109].  The gradient is projected along every authoritative SWMM finite-difference
action direction in that group.  This preserves the fast full-gradient-reuse idea of the
strict audit while allowing smoke/dev model screening before an expensive full checkpoint.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .step2_lazy_stream_v128 import cpu_group_v128_lazy
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _static

V128_DEV_D2_GRADIENT_AUDIT_CONTRACT = (
    "PROJECT7_V128_DEVELOPMENT_D2_GROUP_LOCAL_GRADIENT_AUDIT_V1_FULL_GRADIENT_REUSE"
)


def _full_settings_gradient(
    model: Any,
    *,
    graph: Any,
    static: dict[str, torch.Tensor],
    initial: torch.Tensor,
    rainfall: torch.Tensor,
    flow: torch.Tensor,
    base_sequence: np.ndarray,
    flood_rate_index: int,
) -> np.ndarray:
    settings = torch.as_tensor(
        base_sequence, dtype=initial.dtype, device=initial.device
    ).requires_grad_(True)
    if settings.shape != (72, len(graph.actuator_ids)):
        raise ValueError("V128 development D2 audit action sequence shape mismatch")
    output = model.objective_rollout(
        initial_state=initial,
        rainfall=rainfall,
        settings=settings[None],
        previous_actuator_flow=flow,
        actuator_upstream=static["up"],
        actuator_downstream=static["down"],
        actuator_physics=static["physics"],
        static_node_features=static["static"],
        edge_index=static["edges"],
        flood_rate_index=int(flood_rate_index),
        priority_indices=None,
        dt_seconds=300.0,
    )
    gradient = torch.autograd.grad(output.optimization_tfv_m3.sum(), settings)[0]
    if gradient.shape != settings.shape or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("V128 development D2 audit produced invalid settings gradient")
    return gradient.detach().cpu().numpy().astype(np.float64, copy=False)


def evaluate_d2_gradient_v128_development(
    model: Any,
    *,
    graph: Any,
    base_cache: Any,
    online_cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    device: torch.device,
    flood_rate_index: int = 2,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = [str(name) for name in names]
    if not selected or any(not name.startswith("D2::") for name in selected):
        raise ValueError("development gradient audit requires non-empty held-out D2 groups")
    model.eval().to(device)
    static = _static(graph, device)
    actuator_index = {aid: i for i, aid in enumerate(graph.actuator_ids)}
    rows: list[dict[str, object]] = []
    surrogate_rollouts = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for name in selected:
        entry = base_cache.entry(name)
        arrays = entry.arrays
        required = {
            "actuator_id",
            "requested_setting",
            "base_setting",
            "exact_node_flood_volume_m3",
            "settings",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{name}: D2 cache lacks gradient provenance {missing}")

        cpu = cpu_group_v128_lazy(online_cache, name, normalization)
        initial = cpu["initial"].to(device)
        rainfall_t = cpu["rainfall"].to(device)
        flow = cpu["previous_flow"].to(device)

        group_indices = np.asarray(entry.indices, dtype=np.int64)
        reference_positions = np.flatnonzero(group_indices == int(entry.reference_index))
        if reference_positions.size != 1:
            raise RuntimeError(f"{name}: reference row is not unique inside D2 group")
        reference_position = int(reference_positions[0])
        node_volume = np.asarray(
            arrays["exact_node_flood_volume_m3"][group_indices], dtype=np.float64
        )
        tfv = node_volume.sum(axis=1, dtype=np.float64)
        aids = np.asarray(arrays["actuator_id"])[group_indices].astype(str)
        requested = np.asarray(arrays["requested_setting"][group_indices], dtype=np.float64)
        base_setting = np.asarray(arrays["base_setting"][group_indices], dtype=np.float64)
        sequences = np.asarray(arrays["settings"][group_indices], dtype=np.float32)
        reference_sequence = sequences[reference_position]
        if sequences.shape[1:] != (72, len(graph.actuator_ids)):
            raise ValueError(f"{name}: D2 action-sequence shape mismatch")

        full_gradient = _full_settings_gradient(
            model,
            graph=graph,
            static=static,
            initial=initial,
            rainfall=rainfall_t,
            flow=flow,
            base_sequence=reference_sequence,
            flood_rate_index=int(flood_rate_index),
        )
        surrogate_rollouts += 1

        for aid in sorted(set(aids.tolist()) - {""}):
            if aid not in actuator_index:
                raise ValueError(f"{name}: D2 actuator {aid} absent from graph")
            positions = np.flatnonzero(aids == aid)
            if not positions.size:
                continue
            b_values = base_setting[positions]
            if not np.isfinite(b_values).all():
                raise ValueError(f"{name}/{aid}: D2 base setting is non-finite")
            b = float(b_values[0])
            if not np.allclose(b_values, b, rtol=0.0, atol=1.0e-10):
                raise ValueError(f"{name}/{aid}: D2 base setting drift within group")
            below = positions[requested[positions] < b - 1.0e-10]
            above = positions[requested[positions] > b + 1.0e-10]
            if below.size and above.size:
                lo = int(below[np.argmax(requested[below])])
                hi = int(above[np.argmin(requested[above])])
                du = float(requested[hi] - requested[lo])
                truth = float((tfv[hi] - tfv[lo]) / du)
                direction = (sequences[hi] - sequences[lo]) / du
                method = "central_group_local"
            elif above.size:
                hi = int(above[np.argmin(requested[above])])
                du = float(requested[hi] - b)
                truth = float((tfv[hi] - tfv[reference_position]) / du)
                direction = (sequences[hi] - reference_sequence) / du
                method = "forward_bound_group_local"
            elif below.size:
                lo = int(below[np.argmax(requested[below])])
                du = float(b - requested[lo])
                truth = float((tfv[reference_position] - tfv[lo]) / du)
                direction = (reference_sequence - sequences[lo]) / du
                method = "backward_bound_group_local"
            else:
                continue
            if du <= 1.0e-12 or not np.isfinite(direction).all():
                continue
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1.0e-10:
                continue
            changed = np.flatnonzero(np.max(np.abs(direction), axis=0) > 1.0e-8)
            if set(changed.tolist()) - {actuator_index[aid]}:
                raise ValueError(
                    f"{name}/{aid}: D2 branch changes other actuators; not a single-actuator finite difference"
                )
            predicted = float(np.sum(full_gradient * direction, dtype=np.float64))
            rows.append(
                {
                    "group": name,
                    "rainfall_group": str(entry.rainfall_group),
                    "event_id": str(entry.event_id),
                    "checkpoint_id": str(entry.checkpoint_id),
                    "actuator_id": aid,
                    "base_setting": b,
                    "finite_difference_method": method,
                    "direction_sequence_l2_per_setting": direction_norm,
                    "true_tfv_gradient_m3_per_setting": truth,
                    "predicted_tfv_gradient_m3_per_setting": predicted,
                }
            )
        del cpu, initial, rainfall_t, flow, full_gradient

    if not rows:
        raise RuntimeError("V128 development D2 gradient audit produced no cases")
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["rainfall_group"]), []).append(row)
    group_metrics: list[dict[str, float]] = []
    for _, group in sorted(groups.items()):
        truth = np.asarray([float(row["true_tfv_gradient_m3_per_setting"]) for row in group])
        pred = np.asarray([float(row["predicted_tfv_gradient_m3_per_setting"]) for row in group])
        informative = np.abs(truth) > 1.0e-8
        sign = (
            float(np.mean(np.sign(pred[informative]) == np.sign(truth[informative])))
            if informative.any()
            else float("nan")
        )
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = float(np.dot(truth, pred) / denom) if denom > 1.0e-12 else float("nan")
        group_metrics.append(
            {
                "sign": sign,
                "cosine": cosine,
                "mae": float(np.mean(np.abs(pred - truth))),
            }
        )

    def mean(key: str) -> float | None:
        values = np.asarray([row[key] for row in group_metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    peak_alloc = peak_reserved = 0.0
    if device.type == "cuda":
        gib = float(1024**3)
        peak_alloc = float(torch.cuda.max_memory_allocated(device) / gib)
        peak_reserved = float(torch.cuda.max_memory_reserved(device) / gib)
    metrics: dict[str, object] = {
        "contract": V128_DEV_D2_GRADIENT_AUDIT_CONTRACT,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "gradient_cases": len(rows),
        "gradient_rainfall_groups": len(groups),
        "tfv_gradient_sign_accuracy": mean("sign"),
        "tfv_gradient_cosine_similarity": mean("cosine"),
        "tfv_gradient_mae_m3_per_setting": mean("mae"),
        "predicted_gradient_semantics": (
            "smooth V128 optimization TFV full dJ/dU projected along exact D2 action direction"
        ),
        "truth_gradient_semantics": "authoritative SWMM cumulative TFV group-local finite difference",
        "causal_step1_state": True,
        "causal_rainfall": True,
        "used_for_training": False,
        "surrogate_h360_autograd_rollouts": int(surrogate_rollouts),
        "full_settings_gradient_reused_within_group": True,
        "cuda_peak_allocated_gb": peak_alloc,
        "cuda_peak_reserved_gb": peak_reserved,
    }
    return rows, metrics


__all__ = [
    "V128_DEV_D2_GRADIENT_AUDIT_CONTRACT",
    "evaluate_d2_gradient_v128_development",
]
