"""Freeze an information-efficient Project7 V127 D5 gradient plan; never runs SWMM.

The planner uses development/TrainFit identities, causal Step1 states and engineering
action geometry only.  It deliberately removes two common sources of wasted SWMM work:
near-duplicate hydraulic checkpoints and duplicate/remote random action probes.  Selected
checkpoints are rainfall-balanced and diverse in causal hydraulic/command descriptors;
centres stay on or near the HOLD/Sparse-RBC operating corridor; all directions live in the
exact 12 x 109 online L-BFGS-B fraction tensor and are decoded by the same sequential MPC
transform.  No D5 outcome is read during planning.
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
    deterministic_rainfall_roles_v127,
    directional_fractions_v127,
    json_matrix_v127,
    local_exploration_center_fractions_v127,
    sequence_sha256_v127,
    symmetric_probe_v127,
)
from rtc.production_cli import _load_graph
from rtc.step2_state_store_v127 import load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v60_contract import require_feature
from rtc.step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123
from rtc.step3_mpc_v127 import ContinuousMPCDesignV127, encode_sequence_to_fraction_v127


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _severity(state: np.ndarray, *, depth_index: int, flood_rate_index: int) -> float:
    values = np.asarray(state, dtype=np.float64)
    depth = np.maximum(values[:, int(depth_index)], 0.0)
    flood = np.maximum(values[:, int(flood_rate_index)], 0.0)
    return float(np.quantile(depth, 0.95) + 5.0 * np.quantile(flood, 0.95))


def _descriptor(
    state: np.ndarray,
    current: np.ndarray,
    active_target: np.ndarray,
    *,
    depth_index: int,
    flood_rate_index: int,
) -> np.ndarray:
    """Outcome-blind descriptor used only to avoid redundant D5 checkpoints."""
    values = np.asarray(state, dtype=np.float64)
    depth = np.maximum(values[:, int(depth_index)], 0.0)
    flood = np.maximum(values[:, int(flood_rate_index)], 0.0)
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    active = np.asarray(active_target, dtype=np.float64).reshape(-1)
    lag = np.abs(active - current)
    return np.asarray(
        [
            *np.quantile(depth, [0.50, 0.90, 0.99]).tolist(),
            float(depth.max(initial=0.0)),
            *np.quantile(flood, [0.90, 0.99]).tolist(),
            float(flood.max(initial=0.0)),
            float(np.mean(current)),
            float(np.std(current)),
            float(np.mean(active)),
            float(np.std(active)),
            float(np.mean(lag)),
            float(lag.max(initial=0.0)),
        ],
        dtype=np.float64,
    )


def _farthest_order(values: list[dict[str, object]], mean: np.ndarray, std: np.ndarray) -> list[dict[str, object]]:
    """Order one rainfall group by causal-state novelty, seeded by the severe state."""
    if not values:
        return []
    normalized = [
        (np.asarray(item["descriptor"], dtype=float) - mean) / std for item in values
    ]
    remaining = set(range(len(values)))
    first = max(
        remaining,
        key=lambda i: (float(values[i]["severity"]), str(values[i]["checkpoint_id"])),
    )
    order = [first]
    remaining.remove(first)
    while remaining:
        chosen = max(
            remaining,
            key=lambda i: (
                min(float(np.linalg.norm(normalized[i] - normalized[j])) for j in order),
                float(values[i]["severity"]),
                str(values[i]["checkpoint_id"]),
            ),
        )
        order.append(chosen)
        remaining.remove(chosen)
    return [values[i] for i in order]


def _diverse_balanced_checkpoints(
    records: list[dict[str, object]], maximum: int
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Rainfall-balanced farthest-point selection using no SWMM outcome."""
    if maximum <= 0 or not records:
        return [], {"descriptor_min_distance": float("nan"), "descriptor_median_distance": float("nan")}
    matrix = np.stack([np.asarray(r["descriptor"], dtype=float) for r in records])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 1.0e-9, std, 1.0)
    by_rain: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_rain.setdefault(str(record["rainfall_group"]), []).append(record)
    ordered = {key: _farthest_order(values, mean, std) for key, values in by_rain.items()}
    result: list[dict[str, object]] = []
    seen_identity: set[str] = set()
    rain = sorted(ordered)
    while len(result) < maximum and any(ordered[key] for key in rain):
        for key in rain:
            while ordered[key] and len(result) < maximum:
                item = ordered[key].pop(0)
                identity = hashlib.sha256(
                    np.round(np.asarray(item["descriptor"], dtype=float), 8).tobytes()
                    + str(item["active_target_sha256"]).encode()
                ).hexdigest()
                if identity in seen_identity:
                    continue
                seen_identity.add(identity)
                result.append(item)
                break
    if len(result) < 2:
        distances = np.asarray([], dtype=float)
    else:
        selected = np.stack(
            [(np.asarray(r["descriptor"], dtype=float) - mean) / std for r in result]
        )
        distances = []
        for i in range(len(selected)):
            for j in range(i):
                distances.append(float(np.linalg.norm(selected[i] - selected[j])))
        distances = np.asarray(distances, dtype=float)
    diagnostics = {
        "descriptor_min_distance": float(distances.min()) if distances.size else 0.0,
        "descriptor_median_distance": float(np.median(distances)) if distances.size else 0.0,
    }
    return result, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-checkpoints", type=int, default=24)
    p.add_argument("--directions-per-center", type=int, default=6)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    args = p.parse_args()

    design = D5GradientDesignV127(
        max_checkpoints=int(args.max_checkpoints),
        directions_per_center=int(args.directions_per_center),
    )
    design.validate()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("V127 D5 requires the frozen 109-actuator graph")
    physics_names = tuple(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics, dtype=np.float32)
    min_setting = physics[:, require_feature(physics_names, "min_setting")]
    max_setting = physics[:, require_feature(physics_names, "max_setting")]
    if min_setting.shape != (109,) or max_setting.shape != (109,) or np.any(min_setting > max_setting):
        raise ValueError("V127 D5 graph setting bounds are invalid")

    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2"))
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if (len(fit_d2), len(holdout_d2)) != (112, 32):
        raise ValueError("V127 D5 requires the canonical 112/32 D2 rainfall split")
    state_store = load_causal_state_store_v127(args.causal_state_store)

    records: list[dict[str, object]] = []
    for name in fit_d2:
        entry = cache.entry(name)
        state = state_store.state_for(entry)
        current = state_store.current_setting_for(entry)
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        if reference.shape != (72, 109):
            raise ValueError("V127 D5 reference sequence must be H72 x 109")
        if float(np.max(np.abs(reference[0] - reference[1]))) > 1.0e-7:
            raise ValueError("V127 D5 reference does not preserve the first 10-min target latch")
        active_target = reference[0].copy()
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][entry.reference_index]).reshape(-1)[0])
        records.append(
            {
                "group": name,
                "event_id": str(entry.event_id),
                "rainfall_group": str(entry.rainfall_group),
                "checkpoint_id": str(entry.checkpoint_id),
                "elapsed_seconds": elapsed,
                "severity": _severity(
                    state,
                    depth_index=int(args.depth_index),
                    flood_rate_index=int(args.flood_rate_index),
                ),
                "descriptor": _descriptor(
                    state,
                    current,
                    active_target,
                    depth_index=int(args.depth_index),
                    flood_rate_index=int(args.flood_rate_index),
                ),
                "active_target_sha256": sequence_sha256_v127(active_target),
            }
        )
    selected, selection_diagnostics = _diverse_balanced_checkpoints(records, design.max_checkpoints)
    if not selected:
        raise RuntimeError("V127 D5 could not select any information-bearing checkpoint")
    roles = deterministic_rainfall_roles_v127(
        [str(r["rainfall_group"]) for r in selected], audit_fraction=design.audit_fraction
    )

    mpc_design = ContinuousMPCDesignV127(
        min_improvement_vs_rbc_m3=0.0,
        movement_penalty_m3=0.0,
    )
    rows: list[dict[str, object]] = []
    direction_family_counts: dict[str, int] = {}
    directly_touched = np.zeros((12, 109), dtype=bool)
    skipped_duplicate_centers = 0
    rejected_duplicate_probe_pairs = 0

    for selected_rank, record in enumerate(selected):
        entry = cache.entry(str(record["group"]))
        state = state_store.state_for(entry)
        current = state_store.current_setting_for(entry)
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        active_target = reference[0].copy()
        hold = np.repeat(active_target[None, :], 72, axis=0).astype(np.float32)
        rbc_raw = np.asarray(
            build_sparse_state_auto_rbc_anchor_v123(
                state,
                current,
                hold,
                graph,
                control_block_steps=2,
                max_delta_per_update=0.5,
            ),
            dtype=np.float32,
        )
        hold_fraction = encode_sequence_to_fraction_v127(
            hold,
            active_target=active_target,
            min_setting=min_setting,
            max_setting=max_setting,
            design=mpc_design,
        ).astype(np.float32)
        rbc_fraction = encode_sequence_to_fraction_v127(
            rbc_raw,
            active_target=active_target,
            min_setting=min_setting,
            max_setting=max_setting,
            design=mpc_design,
        ).astype(np.float32)
        identity = f"{entry.rainfall_group}|{entry.event_id}|{entry.checkpoint_id}"
        local_fraction = local_exploration_center_fractions_v127(
            hold_fraction, rbc_fraction, checkpoint_identity=identity
        )
        centers = (
            ("hold", hold_fraction),
            ("rbc_warm_start", rbc_fraction),
            ("local_non_rbc_exploration", local_fraction),
        )

        used_sequence_sha: set[str] = set()
        for center_index, (center_family, center_fraction) in enumerate(centers):
            # Decode centre once before constructing directions, and skip exact duplicate
            # operating points rather than spending another centre + pairs on the same SWMM action.
            preview = symmetric_probe_v127(
                active_target=active_target,
                min_setting=min_setting,
                max_setting=max_setting,
                center_fractions=center_fraction,
                direction=directional_fractions_v127(
                    109,
                    checkpoint_identity=identity,
                    checkpoint_rank=selected_rank,
                    center_index=center_index,
                    direction_index=0,
                    free_control_blocks=design.free_control_blocks,
                )[0],
                design=design,
            )
            if preview is None:
                raise RuntimeError(f"V127 D5 cannot decode centre {identity}/{center_family}")
            center_sequence = np.asarray(preview["center_sequence"], dtype=np.float32)
            center_sha = sequence_sha256_v127(center_sequence)
            if center_sha in used_sequence_sha:
                skipped_duplicate_centers += 1
                continue
            used_sequence_sha.add(center_sha)

            accepted: list[tuple[int, np.ndarray, str, dict[str, object], str, str]] = []
            trial = 0
            while len(accepted) < design.directions_per_center and trial < 128:
                direction, direction_family = directional_fractions_v127(
                    109,
                    checkpoint_identity=identity,
                    checkpoint_rank=selected_rank,
                    center_index=center_index,
                    direction_index=trial,
                    free_control_blocks=design.free_control_blocks,
                )
                probe = symmetric_probe_v127(
                    active_target=active_target,
                    min_setting=min_setting,
                    max_setting=max_setting,
                    center_fractions=center_fraction,
                    direction=direction,
                    design=design,
                )
                trial += 1
                if probe is None:
                    continue
                plus_sha = sequence_sha256_v127(np.asarray(probe["plus_sequence"], dtype=np.float32))
                minus_sha = sequence_sha256_v127(np.asarray(probe["minus_sequence"], dtype=np.float32))
                if plus_sha == minus_sha or plus_sha in used_sequence_sha or minus_sha in used_sequence_sha:
                    rejected_duplicate_probe_pairs += 1
                    continue
                accepted.append((trial - 1, direction, direction_family, probe, plus_sha, minus_sha))
                used_sequence_sha.update((plus_sha, minus_sha))
                directly_touched |= np.abs(direction) > 0.0
                direction_family_counts[direction_family] = direction_family_counts.get(direction_family, 0) + 1
            if len(accepted) != design.directions_per_center:
                raise RuntimeError(
                    f"V127 D5 could not build {design.directions_per_center} unique symmetric directions for {identity}/{center_family}"
                )

            center_fraction_sha = sequence_sha256_v127(center_fraction)
            center_id = hashlib.sha256(
                f"{V127_D5_CONTRACT}|{identity}|{center_family}|{center_fraction_sha}|{center_sha}".encode()
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
                "active_target_json": json_matrix_v127(active_target),
                "center_fractions_json": json_matrix_v127(center_fraction),
                "direct_mpc_variable_shape": "12x109",
                "direct_mpc_variable_count": 12 * 109,
                "control_blocks": 36,
                "free_control_blocks": design.free_control_blocks,
                "model_horizon_steps": 72,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "future_action_rule": "H120_continuous_free_targets_then_hold_terminal_target_to_H360",
                "rbc_role": "warm_start_and_safety_fallback_only",
            }
            rows.append(
                {
                    **common,
                    "plan_row_id": center_id,
                    "probe_role": "center",
                    "direction_id": "",
                    "direction_family": "center",
                    "direction_fractions_json": "",
                    "probe_fractions_json": json_matrix_v127(center_fraction),
                    "epsilon": 0.0,
                    "action_sequence_sha256": center_sha,
                    "action_sequence_json": json_matrix_v127(center_sequence),
                    "midpoint_error": 0.0,
                    "displacement_symmetry_error": 0.0,
                    "physical_displacement_l2": 0.0,
                    "first_move_displacement_l2": 0.0,
                }
            )
            for local_index, (direction_trial, direction, direction_family, probe, plus_sha, minus_sha) in enumerate(accepted):
                direction_id = hashlib.sha256(
                    f"{center_id}|direction|{local_index}|trial{direction_trial}|{direction_family}".encode()
                ).hexdigest()
                for role, seq_key, frac_key, sequence_sha in (
                    ("plus", "plus_sequence", "plus_fractions", plus_sha),
                    ("minus", "minus_sequence", "minus_fractions", minus_sha),
                ):
                    sequence = np.asarray(probe[seq_key], dtype=np.float32)
                    fractions = np.asarray(probe[frac_key], dtype=np.float32)
                    row_id = hashlib.sha256(f"{direction_id}|{role}|{sequence_sha}".encode()).hexdigest()
                    rows.append(
                        {
                            **common,
                            "plan_row_id": row_id,
                            "probe_role": role,
                            "direction_id": direction_id,
                            "direction_family": direction_family,
                            "direction_fractions_json": json_matrix_v127(direction),
                            "probe_fractions_json": json_matrix_v127(fractions),
                            "epsilon": float(probe["epsilon"]),
                            "action_sequence_sha256": sequence_sha,
                            "action_sequence_json": json_matrix_v127(sequence),
                            "midpoint_error": float(probe["midpoint_error"]),
                            "displacement_symmetry_error": float(probe["displacement_symmetry_error"]),
                            "physical_displacement_l2": float(probe["physical_displacement_l2"]),
                            "first_move_displacement_l2": float(probe["first_move_displacement_l2"]),
                        }
                    )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError("V127 D5 planner produced no branches")
    if len(frame) > design.planned_branches:
        raise RuntimeError("V127 D5 exceeded its maximum information-efficient branch budget")
    if frame["plan_row_id"].duplicated().any() or frame["action_sequence_sha256"].isna().any():
        raise RuntimeError("V127 D5 contains duplicate row identities or missing sequences")
    if frame.duplicated(["checkpoint_id", "action_sequence_sha256"]).any():
        raise RuntimeError("V127 D5 still contains duplicate SWMM action sequences at one checkpoint")
    if (frame.groupby("rainfall_group")["split_role"].nunique() != 1).any():
        raise RuntimeError("V127 D5 FIT/AUDIT leaks inside rainfall group")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / "STEP2_V127_D5_GRADIENT_PLAN.csv"
    frame.to_csv(plan, index=False)
    fit_rain = sorted(set(frame.loc[frame["split_role"] == "fit", "rainfall_group"].astype(str)))
    audit_rain = sorted(set(frame.loc[frame["split_role"] == "audit", "rainfall_group"].astype(str)))
    probe_rows = frame[frame["probe_role"] != "center"]
    report = {
        "contract": V127_D5_CONTRACT,
        "verdict": "V127_D5_INFORMATION_VALUE_PLAN_REVIEW_REQUIRED_BEFORE_SWMM",
        "planned_branches": len(frame),
        "maximum_branch_budget": design.planned_branches,
        "selected_checkpoints": len(selected),
        "maximum_checkpoints": design.max_checkpoints,
        "actual_centers": int(frame["center_id"].nunique()),
        "maximum_centers_per_checkpoint": design.centers_per_checkpoint,
        "directions_per_retained_center": design.directions_per_center,
        "gradient_pairs": int((frame["probe_role"] == "plus").sum()),
        "gradient_variable_space": "exact online L-BFGS-B fraction tensor",
        "direct_mpc_variable_shape": [12, 109],
        "direct_mpc_variable_count": 12 * 109,
        "direct_variable_coordinates_touched": int(directly_touched.sum()),
        "actuators_touched": int(np.any(directly_touched, axis=0).sum()),
        "free_blocks_touched": int(np.any(directly_touched, axis=1).sum()),
        "direction_family_counts": direction_family_counts,
        "selection_information_value": selection_diagnostics,
        "skipped_duplicate_centers": skipped_duplicate_centers,
        "rejected_duplicate_probe_pairs": rejected_duplicate_probe_pairs,
        "physical_displacement_l2_median": float(probe_rows["physical_displacement_l2"].median()),
        "first_move_displacement_l2_median": float(probe_rows["first_move_displacement_l2"].median()),
        "actuators": len(graph.actuator_ids),
        "fit_rainfall_groups": fit_rain,
        "audit_rainfall_groups": audit_rain,
        "rainfall_overlap": sorted(set(fit_rain) & set(audit_rain)),
        "center_families": sorted(frame["center_family"].unique().tolist()),
        "epsilon_min": float(frame.loc[frame["epsilon"] > 0, "epsilon"].min()),
        "epsilon_max": float(frame["epsilon"].max()),
        "max_midpoint_error": float(frame["midpoint_error"].max()),
        "max_displacement_symmetry_error": float(frame["displacement_symmetry_error"].max()),
        "lineage": {
            "graph_sha256": _sha(args.graph),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "boundary": {
            "new_swmm": False,
            "selection_uses_d5_outcomes": False,
            "selection_uses_future_hydraulics": False,
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
