"""Audit V12.3 finite-shooting first-move coverage after basis projection."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step3_candidates_v123 import candidate_coefficients_v123, unique_executable_first_moves_v123


def _types(graph) -> np.ndarray:
    names = tuple(graph.actuator_physics_feature_names)
    indices = [names.index(name) for name in ("is_pump", "is_orifice", "is_weir", "is_outlet")]
    return np.asarray(graph.actuator_physics, dtype=np.float64)[:, indices].argmax(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 first-move candidate coverage audit")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    coefficients = candidate_coefficients_v123(basis)
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    type_names = ("pump", "orifice", "weir", "outlet")

    first = coefficients[:, 0, :]
    nonhold = first[1:]
    active_groups = [np.flatnonzero(np.abs(row) > 1e-8) for row in nonhold]
    group_coverage = sorted({int(g) for row in active_groups for g in row})
    group_ids = np.asarray(basis.grouping.group_id_by_actuator, dtype=np.int64)
    actuator_coverage = sorted({int(a) for g in group_coverage for a in np.flatnonzero(group_ids == g)})
    signs = Counter("positive" if float(value) > 0 else "negative" for row in nonhold for value in row[np.abs(row) > 1e-8])
    pair_family = Counter()
    for row in active_groups:
        if len(row) <= 1:
            pair_family["single"] += 1
        else:
            zone_by_group = {
                int(g): int(np.unique(basis.grouping.zone_id_by_actuator[group_ids == g])[0])
                for g in row
            }
            zones = set(zone_by_group.values())
            pair_family["same_zone" if len(zones) == 1 else "cross_zone"] += 1
    type_covered = {}
    actuator_types = _types(graph)
    for kind, label in enumerate(type_names):
        type_covered[label] = int(np.sum(np.isin(np.arange(len(actuator_types)), actuator_coverage) & (actuator_types == kind)))

    per_group: list[dict[str, int]] = []
    for name in names:
        entry = cache.entry(name)
        ref = torch.as_tensor(entry.arrays["settings"][entry.reference_index].copy(), dtype=torch.float32)[None]
        coeff = torch.as_tensor(coefficients, dtype=torch.float32)
        projected = basis.decode(ref.expand(coeff.shape[0], -1, -1), coeff)
        _, report = unique_executable_first_moves_v123(
            projected,
            ref,
            control_block_steps=int(basis.horizon.control_block_steps),
        )
        per_group.append({"group": name, **report})
    unique = np.asarray([r["unique_first_move_count"] for r in per_group], dtype=np.float64)
    passive = np.asarray([r["tail_only_or_passive_like_count"] for r in per_group], dtype=np.float64)
    payload = {
        "contract": "PROJECT7_V123_FIRST_MOVE_COVERAGE_AUDIT_V1",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": hashlib.sha256(Path(args.cache_manifest).read_bytes()).hexdigest(),
        "graph": str(Path(args.graph).resolve()),
        "raw_candidate_count": int(coefficients.shape[0]),
        "raw_nonhold_count": int(nonhold.shape[0]),
        "unique_first_move_count_after_projection": {"min": int(unique.min()), "median": float(np.median(unique)), "mean": float(unique.mean()), "max": int(unique.max())},
        "tail_only_or_passive_like_count_after_projection": {"min": int(passive.min()), "median": float(np.median(passive)), "mean": float(passive.mean()), "max": int(passive.max())},
        "control_groups_total": int(basis.group_count),
        "control_groups_covered_by_nonhold_first_move": int(len(group_coverage)),
        "actuators_total": int(len(graph.actuator_ids)),
        "actuators_covered_by_nonhold_first_move": int(len(actuator_coverage)),
        "actuator_type_coverage": type_covered,
        "direction_counts": dict(signs),
        "first_move_family_counts": dict(pair_family),
        "temporal_basis_zero_active_nonhold": bool(np.all(np.max(np.abs(nonhold), axis=1) > 0.0)),
        "group_reports": per_group,
        "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False},
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "STEP2_V123_FIRST_MOVE_COVERAGE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# STEP2 V123 FIRST-MOVE COVERAGE", "", f"raw candidates: {payload['raw_candidate_count']}", f"unique executable first moves after projection (min/median/mean/max): {unique.min():.0f}/{np.median(unique):.1f}/{unique.mean():.1f}/{unique.max():.0f}", f"tail-only/passive-like after projection (min/median/mean/max): {passive.min():.0f}/{np.median(passive):.1f}/{passive.mean():.1f}/{passive.max():.0f}", f"groups covered: {len(group_coverage)}/{basis.group_count}", f"actuators covered: {len(actuator_coverage)}/{len(graph.actuator_ids)}", f"direction counts: {dict(signs)}", f"families: {dict(pair_family)}", ""]
    (out / "STEP2_V123_FIRST_MOVE_COVERAGE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
