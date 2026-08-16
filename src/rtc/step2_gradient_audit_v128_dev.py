"""Development-only D2 TFV-gradient audit for nonfinal V128 stage checkpoints.

For each held-out D2 group the exact reference H360 action sequence is differentiated once to
obtain full dJ/dU[72,109].  Every authoritative SWMM candidate is then audited along its exact
reference-to-candidate action-sequence direction.  The D2 cache already freezes those action
sequences in reference-then-candidate order, so row-level ``actuator_id``, ``base_setting`` and
``requested_setting`` metadata are not required and must never be synthesised.

The magnitude diagnostic divides both the authoritative TFV change and the autograd projection
by the candidate's peak absolute setting displacement.  This is a directional sensitivity along
the *actual D2 pulse shape*, not a claim that all 72 time steps share one scalar setting.  D2 is
contractually single-actuator; zero-change or multi-actuator candidates fail closed.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .step2_lazy_stream_v128 import cpu_group_v128_lazy
from .step2_spatial_audit_v128 import _changed_actuator_per_candidate
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _static, _truth_node_volume

V128_DEV_D2_GRADIENT_AUDIT_CONTRACT = (
    "PROJECT7_V128_DEVELOPMENT_D2_SETTINGS_DERIVED_DIRECTIONAL_GRADIENT_AUDIT_V2"
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


def _candidate_direction_from_settings(
    reference_sequence: np.ndarray,
    candidate_sequence: np.ndarray,
    *,
    expected_actuator_index: int,
    tolerance: float = 1.0e-7,
) -> tuple[np.ndarray, float, int]:
    """Return exact D2 direction normalised by its peak absolute setting displacement.

    ``direction`` retains the physical sign and full temporal pulse/ramp shape.  Its changed
    actuator has peak absolute magnitude one.  This construction is lossless with respect to
    the stored reference/candidate settings and avoids inventing missing scalar provenance.
    """
    reference = np.asarray(reference_sequence, dtype=np.float64)
    candidate = np.asarray(candidate_sequence, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("D2 gradient direction requires matching [H,actuator] settings")
    delta = candidate - reference
    if not np.isfinite(delta).all():
        raise ValueError("D2 gradient direction contains non-finite settings")
    changed = np.flatnonzero(np.max(np.abs(delta), axis=0) > float(tolerance))
    if changed.size != 1:
        raise ValueError(
            f"D2 gradient candidate changes {changed.size} actuators; expected exactly one"
        )
    actuator_index = int(changed[0])
    if actuator_index != int(expected_actuator_index):
        raise RuntimeError("D2 changed-actuator inference disagrees across audit helpers")
    trace = delta[:, actuator_index]
    peak_abs = float(np.max(np.abs(trace)))
    if not np.isfinite(peak_abs) or peak_abs <= float(tolerance):
        raise ValueError("D2 gradient candidate has no finite nonzero setting displacement")
    active_steps = int(np.count_nonzero(np.abs(trace) > float(tolerance)))
    if active_steps <= 0:
        raise ValueError("D2 gradient candidate has no active perturbation steps")
    direction = delta / peak_abs
    if not np.isfinite(direction).all():
        raise ValueError("D2 normalised gradient direction is non-finite")
    return direction, peak_abs, active_steps


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
    rows: list[dict[str, object]] = []
    surrogate_rollouts = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for name in selected:
        entry = base_cache.entry(name)
        cpu = cpu_group_v128_lazy(online_cache, name, normalization)
        initial = cpu["initial"].to(device)
        rainfall_t = cpu["rainfall"].to(device)
        flow = cpu["previous_flow"].to(device)
        sequences = np.asarray(cpu["settings"], dtype=np.float32)
        if sequences.ndim != 3 or sequences.shape[1:] != (72, len(graph.actuator_ids)):
            raise ValueError(f"{name}: D2 action-sequence shape mismatch")
        if sequences.shape[0] < 2:
            raise ValueError(f"{name}: D2 gradient audit requires reference plus candidate")

        # Both helpers use the same authoritative reference-first action sequence.  The spatial
        # audit already relies on this invariant, so gradient screening should not demand a
        # second, lossy row-level provenance schema that the canonical cache never promised.
        actuator_indices = _changed_actuator_per_candidate(cpu["settings"])
        node_volume = _truth_node_volume(base_cache, name).astype(np.float64, copy=False)
        if node_volume.shape[0] != sequences.shape[0]:
            raise RuntimeError(f"{name}: D2 gradient truth/action branch order mismatch")
        tfv = node_volume.sum(axis=1, dtype=np.float64)
        reference_sequence = sequences[0]

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

        for candidate_position in range(1, sequences.shape[0]):
            actuator_idx = int(actuator_indices[candidate_position - 1])
            if actuator_idx < 0 or actuator_idx >= len(graph.actuator_ids):
                raise RuntimeError(f"{name}: inferred D2 actuator index is outside graph")
            direction, peak_step, active_steps = _candidate_direction_from_settings(
                reference_sequence,
                sequences[candidate_position],
                expected_actuator_index=actuator_idx,
            )
            true_delta = float(tfv[candidate_position] - tfv[0])
            truth = true_delta / peak_step
            predicted = float(np.sum(full_gradient * direction, dtype=np.float64))
            if not np.isfinite(truth) or not np.isfinite(predicted):
                raise RuntimeError(f"{name}: D2 directional-gradient case is non-finite")
            rows.append(
                {
                    "group": name,
                    "rainfall_group": str(entry.rainfall_group),
                    "event_id": str(entry.event_id),
                    "checkpoint_id": str(entry.checkpoint_id),
                    "candidate_position": int(candidate_position),
                    "actuator_index": actuator_idx,
                    "actuator_id": str(graph.actuator_ids[actuator_idx]),
                    "finite_difference_method": "reference_to_candidate_exact_sequence_direction",
                    "peak_setting_step_abs": peak_step,
                    "active_perturbation_model_steps": active_steps,
                    "direction_sequence_l2_per_peak_setting": float(np.linalg.norm(direction)),
                    "true_tfv_delta_m3": true_delta,
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
        truth = np.asarray(
            [float(row["true_tfv_gradient_m3_per_setting"]) for row in group],
            dtype=np.float64,
        )
        pred = np.asarray(
            [float(row["predicted_tfv_gradient_m3_per_setting"]) for row in group],
            dtype=np.float64,
        )
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
    mae = mean("mae")
    metrics: dict[str, object] = {
        "contract": V128_DEV_D2_GRADIENT_AUDIT_CONTRACT,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "gradient_cases": len(rows),
        "gradient_rainfall_groups": len(groups),
        "tfv_gradient_sign_accuracy": mean("sign"),
        "tfv_gradient_cosine_similarity": mean("cosine"),
        # Compatibility name retained for existing reporting.  The exact unit/semantics are
        # made explicit below and are not fabricated from absent requested/base-setting fields.
        "tfv_gradient_mae_m3_per_setting": mae,
        "tfv_gradient_mae_m3_per_peak_setting_step": mae,
        "gradient_variable_space": "exact stored H72x109 D2 setting-sequence direction",
        "gradient_magnitude_normalization": (
            "candidate peak absolute setting displacement; direction retains exact temporal shape"
        ),
        "predicted_gradient_semantics": (
            "smooth V128 optimization TFV full dJ/dU projected along exact stored D2 "
            "reference-to-candidate action direction"
        ),
        "truth_gradient_semantics": (
            "authoritative SWMM cumulative TFV reference-to-candidate finite difference "
            "per peak absolute setting displacement"
        ),
        "gradient_provenance": (
            "losslessly derived from canonical reference-first settings sequences; no synthetic "
            "actuator_id/base_setting/requested_setting metadata"
        ),
        "single_actuator_action_contract_verified": True,
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
    "_candidate_direction_from_settings",
    "evaluate_d2_gradient_v128_development",
]
