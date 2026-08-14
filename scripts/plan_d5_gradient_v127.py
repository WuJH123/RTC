"""Freeze Project7 V127 D5 antithetic gradient probes; never runs SWMM.

The plan uses only development/TrainFit checkpoint identities, causal Step1 states,
engineering geometry and deterministic action bases.  SWMM outcomes are never inspected
for state selection, action-centre construction or FIT/AUDIT assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.d5_gradient_v127 import (
    D5GradientDesignV127,
    V127_D5_CONTRACT,
    broad_center_coefficients_v127,
    deterministic_rainfall_roles_v127,
    directional_coefficients_v127,
    json_matrix_v127,
    sequence_sha256_v127,
    symmetric_probe_v127,
)
from rtc.production_cli import _load_graph
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_state_store_v127 import load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _severity(state: np.ndarray, *, depth_index: int, flood_rate_index: int) -> float:
    values = np.asarray(state, dtype=np.float64)
    depth = np.maximum(values[:, int(depth_index)], 0.0)
    flood = np.maximum(values[:, int(flood_rate_index)], 0.0)
    return float(np.quantile(depth, 0.95) + 5.0 * np.quantile(flood, 0.95))


def _balanced_checkpoints(records: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    """Rainfall-balanced mixture of high, mid and low current hydraulic severity."""
    by_rain: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_rain.setdefault(str(record["rainfall_group"]), []).append(record)
    for values in by_rain.values():
        values.sort(key=lambda r: (float(r["severity"]), str(r["checkpoint_id"])))
        if len(values) > 2:
            mid = len(values) // 2
            ordered: list[dict[str, object]] = []
            lo, hi = 0, len(values) - 1
            used: set[int] = set()
            for idx in (hi, mid, lo):
                if idx not in used:
                    ordered.append(values[idx]); used.add(idx)
            offset = 1
            while len(used) < len(values):
                for idx in (hi - offset, mid + offset, mid - offset, lo + offset):
                    if 0 <= idx < len(values) and idx not in used:
                        ordered.append(values[idx]); used.add(idx)
                offset += 1
            values[:] = ordered
    result: list[dict[str, object]] = []
    rain = sorted(by_rain)
    while len(result) < maximum and any(by_rain[key] for key in rain):
        for key in rain:
            if by_rain[key] and len(result) < maximum:
                result.append(by_rain[key].pop(0))
    return result


def _coeff_clip(value: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float32), -float(limit), float(limit))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-checkpoints", type=int, default=48)
    p.add_argument("--directions-per-center", type=int, default=4)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    args = p.parse_args()

    design = D5GradientDesignV127(
        max_checkpoints=int(args.max_checkpoints),
        directions_per_center=int(args.directions_per_center),
    )
    design.validate()
    graph = _load_graph(args.graph)
    basis = build_control_basis_v60(graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("V127 D5 requires the frozen 109-actuator graph")
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2"))
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if (len(fit_d2), len(holdout_d2)) != (112, 32):
        raise ValueError("V127 D5 requires the frozen 112/32 D2 rainfall split")
    state_store = load_causal_state_store_v127(args.causal_state_store)

    records: list[dict[str, object]] = []
    for name in fit_d2:
        entry = cache.entry(name)
        state = state_store.state_for(entry)
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][entry.reference_index]).reshape(-1)[0])
        records.append({
            "group": name,
            "event_id": str(entry.event_id),
            "rainfall_group": str(entry.rainfall_group),
            "checkpoint_id": str(entry.checkpoint_id),
            "elapsed_seconds": elapsed,
            "severity": _severity(
                state, depth_index=int(args.depth_index), flood_rate_index=int(args.flood_rate_index)
            ),
        })
    selected = _balanced_checkpoints(records, design.max_checkpoints)
    if len(selected) != design.max_checkpoints:
        raise RuntimeError("V127 D5 could not select the frozen checkpoint budget")
    roles = deterministic_rainfall_roles_v127(
        [str(r["rainfall_group"]) for r in selected], audit_fraction=design.audit_fraction
    )

    rows: list[dict[str, object]] = []
    coefficient_limit = float(basis.contract.coefficient_limit)
    for selected_rank, record in enumerate(selected):
        entry = cache.entry(str(record["group"]))
        state = state_store.state_for(entry)
        current = state_store.current_setting_for(entry)
        # Active target/reference comes from the exact branch design, not realised current
        # position.  This preserves the target-latch distinction used by online RTC.
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        if reference.shape != (72, 109):
            raise ValueError("V127 D5 reference sequence must be H72 x 109")
        reference = np.repeat(reference[::2], 2, axis=0).astype(np.float32)
        rbc = np.asarray(
            build_sparse_state_auto_rbc_anchor_v123(
                state,
                current,
                reference,
                graph,
                control_block_steps=2,
                max_delta_per_update=0.5,
            ),
            dtype=np.float32,
        )
        rbc_coeff = basis.project_actions_to_coefficients(rbc - reference).reshape(
            basis.temporal_basis_count, basis.group_count
        )
        identity = f"{entry.rainfall_group}|{entry.event_id}|{entry.checkpoint_id}"
        centers = (
            ("hold", np.zeros((basis.temporal_basis_count, basis.group_count), dtype=np.float32)),
            ("rbc_warm_start", _coeff_clip(rbc_coeff, coefficient_limit)),
            ("broad_manifold", _coeff_clip(broad_center_coefficients_v127(basis, checkpoint_identity=identity), coefficient_limit)),
        )
        for center_index, (center_family, center_coeff) in enumerate(centers):
            accepted: list[tuple[int, np.ndarray, dict[str, object]]] = []
            trial = 0
            while len(accepted) < design.directions_per_center and trial < 32:
                direction = directional_coefficients_v127(
                    basis,
                    checkpoint_identity=identity,
                    center_index=center_index,
                    direction_index=trial,
                )
                probe = symmetric_probe_v127(
                    basis,
                    reference_sequence=reference,
                    center_coefficients=center_coeff,
                    direction=direction,
                    design=design,
                )
                if probe is not None:
                    accepted.append((trial, direction, probe))
                trial += 1
            if len(accepted) != design.directions_per_center:
                raise RuntimeError(
                    f"V127 D5 could not build {design.directions_per_center} symmetric directions for {identity}/{center_family}"
                )
            center_sequence = np.asarray(accepted[0][2]["center_sequence"], dtype=np.float32)
            center_sha = sequence_sha256_v127(center_sequence)
            center_id = hashlib.sha256(
                f"{V127_D5_CONTRACT}|{identity}|{center_family}|{center_sha}".encode()
            ).hexdigest()
            common = {
                "contract": V127_D5_CONTRACT,
                "selected_rank": selected_rank,
                "split_role": roles[str(entry.rainfall_group)],
                "event_id": str(entry.event_id),
                "rainfall_group": str(entry.rainfall_group),
                "checkpoint_id": str(entry.checkpoint_id),
                "elapsed_seconds": int(record["elapsed_seconds"]),
                "severity": float(record["severity"]),
                "center_id": center_id,
                "center_family": center_family,
                "center_coefficients_json": json_matrix_v127(center_coeff),
                "control_blocks": 36,
                "free_control_blocks": design.free_control_blocks,
                "model_horizon_steps": 72,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "future_action_rule": "H120_continuous_free_targets_then_hold_terminal_target_to_H360",
                "rbc_role": "warm_start_and_safety_fallback_only",
            }
            rows.append({
                **common,
                "plan_row_id": center_id,
                "probe_role": "center",
                "direction_id": "",
                "direction_coefficients_json": "",
                "epsilon": 0.0,
                "action_sequence_sha256": center_sha,
                "action_sequence_json": json_matrix_v127(center_sequence),
                "midpoint_error": 0.0,
                "displacement_symmetry_error": 0.0,
            })
            for local_index, (direction_trial, direction, probe) in enumerate(accepted):
                direction_id = hashlib.sha256(
                    f"{center_id}|direction|{local_index}|trial{direction_trial}".encode()
                ).hexdigest()
                for role, key in (("plus", "plus_sequence"), ("minus", "minus_sequence")):
                    sequence = np.asarray(probe[key], dtype=np.float32)
                    sequence_sha = sequence_sha256_v127(sequence)
                    row_id = hashlib.sha256(
                        f"{direction_id}|{role}|{sequence_sha}".encode()
                    ).hexdigest()
                    rows.append({
                        **common,
                        "plan_row_id": row_id,
                        "probe_role": role,
                        "direction_id": direction_id,
                        "direction_coefficients_json": json_matrix_v127(direction),
                        "epsilon": float(probe["epsilon"]),
                        "action_sequence_sha256": sequence_sha,
                        "action_sequence_json": json_matrix_v127(sequence),
                        "midpoint_error": float(probe["midpoint_error"]),
                        "displacement_symmetry_error": float(probe["displacement_symmetry_error"]),
                    })

    frame = pd.DataFrame.from_records(rows)
    expected = design.planned_branches
    if len(frame) != expected:
        raise RuntimeError(f"V127 D5 branch count {len(frame)} != frozen {expected}")
    if frame["plan_row_id"].duplicated().any() or frame["action_sequence_sha256"].isna().any():
        raise RuntimeError("V127 D5 contains duplicate row identities or missing sequences")
    if (frame.groupby("rainfall_group")["split_role"].nunique() != 1).any():
        raise RuntimeError("V127 D5 FIT/AUDIT leaks inside rainfall group")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / "STEP2_V127_D5_GRADIENT_PLAN.csv"
    frame.to_csv(plan, index=False)
    fit_rain = sorted(set(frame.loc[frame["split_role"] == "fit", "rainfall_group"].astype(str)))
    audit_rain = sorted(set(frame.loc[frame["split_role"] == "audit", "rainfall_group"].astype(str)))
    report = {
        "contract": V127_D5_CONTRACT,
        "verdict": "V127_D5_PLAN_FROZEN_REVIEW_REQUIRED_BEFORE_SWMM",
        "planned_branches": len(frame),
        "selected_checkpoints": len(selected),
        "centers_per_checkpoint": design.centers_per_checkpoint,
        "directions_per_center": design.directions_per_center,
        "gradient_pairs": int((frame["probe_role"] == "plus").sum()),
        "coefficient_dimension": basis.coefficient_dimension,
        "actuators": len(graph.actuator_ids),
        "fit_rainfall_groups": fit_rain,
        "audit_rainfall_groups": audit_rain,
        "rainfall_overlap": sorted(set(fit_rain) & set(audit_rain)),
        "center_families": sorted(frame["center_family"].unique().tolist()),
        "epsilon_min": float(frame.loc[frame["epsilon"] > 0, "epsilon"].min()),
        "epsilon_max": float(frame["epsilon"].max()),
        "max_midpoint_error": float(frame["midpoint_error"].max()),
        "max_displacement_symmetry_error": float(frame["displacement_symmetry_error"].max()),
        "control_basis": basis_manifest_v60(basis),
        "lineage": {
            "graph_sha256": _sha(args.graph),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "boundary": {
            "new_swmm": False,
            "selection_uses_future_outcomes": False,
            "rbc_is_action_space_ceiling": False,
            "rbc_is_value_reference": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
        "plan_csv": str(plan.resolve()),
    }
    report_path = out / "STEP2_V127_D5_GRADIENT_PLAN.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
