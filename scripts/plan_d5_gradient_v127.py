"""Freeze an information-efficient Project7 V127 D5 gradient plan; never runs SWMM.

Selection is outcome-blind.  Checkpoints are rainfall-balanced and diverse in causal
hydraulic/command descriptors.  Centres remain in operating regions the rolling controller
can plausibly visit (HOLD, Sparse-RBC and local non-RBC exploration).  Gradient probes use
the exact online 12 x 109 fraction variables and exact online sequential decoder.  Centre
executability is checked independently from +/- probe symmetry so a valid boundary centre
is not rejected merely because one trial direction cannot support a central difference.
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
    decode_fractions_v127,
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


def _severity(state: np.ndarray, depth_index: int, flood_rate_index: int) -> float:
    values = np.asarray(state, dtype=np.float64)
    depth = np.maximum(values[:, depth_index], 0.0)
    flood = np.maximum(values[:, flood_rate_index], 0.0)
    return float(np.quantile(depth, 0.95) + 5.0 * np.quantile(flood, 0.95))


def _descriptor(
    state: np.ndarray,
    current: np.ndarray,
    active: np.ndarray,
    depth_index: int,
    flood_rate_index: int,
) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    depth = np.maximum(values[:, depth_index], 0.0)
    flood = np.maximum(values[:, flood_rate_index], 0.0)
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    active = np.asarray(active, dtype=np.float64).reshape(-1)
    lag = np.abs(active - current)
    return np.asarray(
        [
            *np.quantile(depth, [0.50, 0.90, 0.99]),
            depth.max(initial=0.0),
            *np.quantile(flood, [0.90, 0.99]),
            flood.max(initial=0.0),
            current.mean(),
            current.std(),
            active.mean(),
            active.std(),
            lag.mean(),
            lag.max(initial=0.0),
        ],
        dtype=np.float64,
    )


def _farthest_order(
    values: list[dict[str, object]], mean: np.ndarray, std: np.ndarray
) -> list[dict[str, object]]:
    if not values:
        return []
    normalized = [(np.asarray(v["descriptor"], dtype=float) - mean) / std for v in values]
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


def _select_checkpoints(
    records: list[dict[str, object]], maximum: int
) -> tuple[list[dict[str, object]], dict[str, float]]:
    if maximum <= 0 or not records:
        raise ValueError("V127 D5 checkpoint budget/source is empty")
    matrix = np.stack([np.asarray(r["descriptor"], dtype=float) for r in records])
    mean = matrix.mean(axis=0)
    std = np.where(matrix.std(axis=0) > 1.0e-9, matrix.std(axis=0), 1.0)
    by_rain: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_rain.setdefault(str(record["rainfall_group"]), []).append(record)
    ordered = {k: _farthest_order(v, mean, std) for k, v in by_rain.items()}
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    rain = sorted(ordered)
    while len(selected) < maximum and any(ordered[k] for k in rain):
        for key in rain:
            while ordered[key] and len(selected) < maximum:
                item = ordered[key].pop(0)
                identity = hashlib.sha256(
                    np.round(np.asarray(item["descriptor"], dtype=float), 8).tobytes()
                    + str(item["active_target_sha256"]).encode()
                ).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(item)
                break
    normalized = [
        (np.asarray(r["descriptor"], dtype=float) - mean) / std for r in selected
    ]
    distances = np.asarray(
        [
            float(np.linalg.norm(normalized[i] - normalized[j]))
            for i in range(len(normalized))
            for j in range(i)
        ],
        dtype=float,
    )
    return selected, {
        "descriptor_min_pair_distance": float(distances.min()) if distances.size else 0.0,
        "descriptor_median_pair_distance": float(np.median(distances)) if distances.size else 0.0,
    }


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
        raise ValueError("V127 D5 requires 109 actuators")
    physics = np.asarray(graph.actuator_physics, dtype=np.float32)
    names = tuple(graph.actuator_physics_feature_names)
    lo = physics[:, require_feature(names, "min_setting")]
    hi = physics[:, require_feature(names, "max_setting")]
    if lo.shape != (109,) or hi.shape != (109,) or np.any(lo > hi):
        raise ValueError("V127 D5 graph setting bounds are invalid")

    cache = V60TrainCache(args.cache_manifest)
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(cache.names("D2")), holdout_fraction=0.20
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    if (len(fit_d2), len(hold_d2)) != (112, 32):
        raise ValueError("V127 D5 requires canonical D2 112/32 split")
    state_store = load_causal_state_store_v127(args.causal_state_store)

    records: list[dict[str, object]] = []
    for name in fit_d2:
        entry = cache.entry(name)
        state = state_store.state_for(entry)
        current = state_store.current_setting_for(entry)
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        if reference.shape != (72, 109) or not np.allclose(
            reference[0], reference[1], rtol=0.0, atol=1.0e-7
        ):
            raise ValueError("V127 D5 reference does not preserve 10-min first target")
        active = reference[0]
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][entry.reference_index]).reshape(-1)[0])
        records.append(
            {
                "group": name,
                "event_id": str(entry.event_id),
                "rainfall_group": str(entry.rainfall_group),
                "checkpoint_id": str(entry.checkpoint_id),
                "elapsed_seconds": elapsed,
                "severity": _severity(state, args.depth_index, args.flood_rate_index),
                "descriptor": _descriptor(
                    state, current, active, args.depth_index, args.flood_rate_index
                ),
                "active_target_sha256": sequence_sha256_v127(active),
            }
        )
    selected, selection_diag = _select_checkpoints(records, design.max_checkpoints)
    if not selected:
        raise RuntimeError("V127 D5 selected no information-bearing checkpoint")
    roles = deterministic_rainfall_roles_v127(
        [str(r["rainfall_group"]) for r in selected], audit_fraction=design.audit_fraction
    )

    mpc_design = ContinuousMPCDesignV127(
        min_improvement_vs_rbc_m3=0.0, movement_penalty_m3=0.0
    )
    rows: list[dict[str, object]] = []
    direction_counts: dict[str, int] = {}
    touched = np.zeros((12, 109), dtype=bool)
    skipped_centres = rejected_pairs = 0

    for selected_rank, record in enumerate(selected):
        entry = cache.entry(str(record["group"]))
        state = state_store.state_for(entry)
        current = state_store.current_setting_for(entry)
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        active = reference[0].copy()
        hold = np.repeat(active[None, :], 72, axis=0).astype(np.float32)
        rbc = np.asarray(
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
            active_target=active,
            min_setting=lo,
            max_setting=hi,
            design=mpc_design,
        ).astype(np.float32)
        rbc_fraction = encode_sequence_to_fraction_v127(
            rbc,
            active_target=active,
            min_setting=lo,
            max_setting=hi,
            design=mpc_design,
        ).astype(np.float32)
        identity = f"{entry.rainfall_group}|{entry.event_id}|{entry.checkpoint_id}"
        local_fraction = local_exploration_center_fractions_v127(
            hold_fraction, rbc_fraction, checkpoint_identity=identity
        )
        centres = (
            ("hold", hold_fraction),
            ("rbc_warm_start", rbc_fraction),
            ("local_non_rbc_exploration", local_fraction),
        )
        used_sequences: set[str] = set()

        for centre_index, (centre_family, centre_fraction) in enumerate(centres):
            centre_sequence = decode_fractions_v127(
                fractions=centre_fraction,
                active_target=active,
                min_setting=lo,
                max_setting=hi,
                design=design,
            )
            centre_sha = sequence_sha256_v127(centre_sequence)
            if centre_sha in used_sequences:
                skipped_centres += 1
                continue
            used_sequences.add(centre_sha)

            accepted: list[tuple[int, np.ndarray, str, dict[str, object], str, str]] = []
            for trial in range(128):
                if len(accepted) >= design.directions_per_center:
                    break
                direction, family = directional_fractions_v127(
                    109,
                    checkpoint_identity=identity,
                    checkpoint_rank=selected_rank,
                    center_index=centre_index,
                    direction_index=trial,
                    free_control_blocks=design.free_control_blocks,
                )
                probe = symmetric_probe_v127(
                    active_target=active,
                    min_setting=lo,
                    max_setting=hi,
                    center_fractions=centre_fraction,
                    direction=direction,
                    design=design,
                )
                if probe is None:
                    continue
                plus_sha = sequence_sha256_v127(np.asarray(probe["plus_sequence"], dtype=np.float32))
                minus_sha = sequence_sha256_v127(np.asarray(probe["minus_sequence"], dtype=np.float32))
                if plus_sha == minus_sha or plus_sha in used_sequences or minus_sha in used_sequences:
                    rejected_pairs += 1
                    continue
                used_sequences.update((plus_sha, minus_sha))
                accepted.append((trial, direction, family, probe, plus_sha, minus_sha))
                touched |= np.abs(direction) > 0.0
                direction_counts[family] = direction_counts.get(family, 0) + 1
            if len(accepted) != design.directions_per_center:
                raise RuntimeError(
                    f"V127 D5 lacks {design.directions_per_center} unique symmetric probes for {identity}/{centre_family}"
                )

            centre_id = hashlib.sha256(
                f"{V127_D5_CONTRACT}|{identity}|{centre_family}|{sequence_sha256_v127(centre_fraction)}|{centre_sha}".encode()
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
                "center_id": centre_id,
                "center_family": centre_family,
                "active_target_json": json_matrix_v127(active),
                "center_fractions_json": json_matrix_v127(centre_fraction),
                "direct_mpc_variable_shape": "12x109",
                "direct_mpc_variable_count": 1308,
                "control_blocks": 36,
                "free_control_blocks": 12,
                "model_horizon_steps": 72,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "future_action_rule": "H120_continuous_free_targets_then_hold_terminal_target_to_H360",
                "rbc_role": "warm_start_and_safety_fallback_only",
            }
            rows.append(
                {
                    **common,
                    "plan_row_id": centre_id,
                    "probe_role": "center",
                    "direction_id": "",
                    "direction_family": "center",
                    "direction_fractions_json": "",
                    "probe_fractions_json": json_matrix_v127(centre_fraction),
                    "epsilon": 0.0,
                    "action_sequence_sha256": centre_sha,
                    "action_sequence_json": json_matrix_v127(centre_sequence),
                    "midpoint_error": 0.0,
                    "displacement_symmetry_error": 0.0,
                    "physical_displacement_l2": 0.0,
                    "first_move_displacement_l2": 0.0,
                }
            )
            for local_index, (trial, direction, family, probe, plus_sha, minus_sha) in enumerate(accepted):
                direction_id = hashlib.sha256(
                    f"{centre_id}|{local_index}|{trial}|{family}".encode()
                ).hexdigest()
                for role, sequence_key, fraction_key, sequence_sha in (
                    ("plus", "plus_sequence", "plus_fractions", plus_sha),
                    ("minus", "minus_sequence", "minus_fractions", minus_sha),
                ):
                    rows.append(
                        {
                            **common,
                            "plan_row_id": hashlib.sha256(
                                f"{direction_id}|{role}|{sequence_sha}".encode()
                            ).hexdigest(),
                            "probe_role": role,
                            "direction_id": direction_id,
                            "direction_family": family,
                            "direction_fractions_json": json_matrix_v127(direction),
                            "probe_fractions_json": json_matrix_v127(np.asarray(probe[fraction_key])),
                            "epsilon": float(probe["epsilon"]),
                            "action_sequence_sha256": sequence_sha,
                            "action_sequence_json": json_matrix_v127(np.asarray(probe[sequence_key])),
                            "midpoint_error": float(probe["midpoint_error"]),
                            "displacement_symmetry_error": float(probe["displacement_symmetry_error"]),
                            "physical_displacement_l2": float(probe["physical_displacement_l2"]),
                            "first_move_displacement_l2": float(probe["first_move_displacement_l2"]),
                        }
                    )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty or len(frame) > design.planned_branches:
        raise RuntimeError("V127 D5 branch count is empty or exceeds information budget")
    if frame["plan_row_id"].duplicated().any():
        raise RuntimeError("V127 D5 duplicates plan row identities")
    if frame.duplicated(["checkpoint_id", "action_sequence_sha256"]).any():
        raise RuntimeError("V127 D5 contains duplicate SWMM actions at one checkpoint")
    if (frame.groupby("rainfall_group")["split_role"].nunique() != 1).any():
        raise RuntimeError("V127 D5 FIT/AUDIT rainfall leakage")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "STEP2_V127_D5_GRADIENT_PLAN.csv"
    frame.to_csv(csv_path, index=False)
    probes = frame[frame["probe_role"] != "center"]
    fit_rain = sorted(set(frame.loc[frame["split_role"] == "fit", "rainfall_group"].astype(str)))
    audit_rain = sorted(set(frame.loc[frame["split_role"] == "audit", "rainfall_group"].astype(str)))
    report = {
        "contract": V127_D5_CONTRACT,
        "verdict": "V127_D5_INFORMATION_VALUE_PLAN_REVIEW_REQUIRED_BEFORE_SWMM",
        "planned_branches": int(len(frame)),
        "maximum_branch_budget": int(design.planned_branches),
        "selected_checkpoints": int(len(selected)),
        "maximum_checkpoints": int(design.max_checkpoints),
        "actual_centers": int(frame["center_id"].nunique()),
        "directions_per_retained_center": int(design.directions_per_center),
        "gradient_pairs": int((frame["probe_role"] == "plus").sum()),
        "gradient_variable_space": "exact online 12x109 L-BFGS-B fraction tensor",
        "direct_mpc_variable_count": 1308,
        "direct_variable_coordinates_touched": int(touched.sum()),
        "actuators_touched": int(np.any(touched, axis=0).sum()),
        "free_blocks_touched": int(np.any(touched, axis=1).sum()),
        "direction_family_counts": direction_counts,
        "selection_information_value": selection_diag,
        "skipped_duplicate_centers": int(skipped_centres),
        "rejected_duplicate_probe_pairs": int(rejected_pairs),
        "physical_displacement_l2_median": float(probes["physical_displacement_l2"].median()),
        "first_move_displacement_l2_median": float(probes["first_move_displacement_l2"].median()),
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
        "plan_csv": str(csv_path.resolve()),
    }
    (out / "STEP2_V127_D5_GRADIENT_PLAN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
