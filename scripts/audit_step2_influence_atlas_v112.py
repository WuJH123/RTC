"""Build a no-SWMM, TrainFit-only D2 influence-support atlas for V11.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_d2_lineage_v112 import classify_d2_population_v112
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_influence_prior_v112 import InfluencePriorAccumulatorV112
from rtc.step2_influence_support_v112 import (
    build_influence_support_targets_v112,
    infer_single_changed_actuator_v112,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v110_contract import HydraulicHorizonV110
from run_step2_v110 import _load_graph

CONTRACT = "PROJECT7_STEP2_V112_TRAINFIT_INFLUENCE_ATLAS_AUDIT_V1"


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _branches(cache: V60TrainCache, names: list[str]) -> int:
    return int(sum(len(cache.entry(name).indices) for name in names))


def _unique(cache: V60TrainCache, names: list[str], attr: str) -> int:
    return len({str(getattr(cache.entry(name), attr)) for name in names})


def _load(entry, max_depth: np.ndarray) -> float:
    state = np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float64)
    valid = np.isfinite(state[:, 0]) & np.isfinite(max_depth) & (max_depth > 1e-6)
    if not valid.any():
        raise ValueError("V112 checkpoint has no valid depth/capacity nodes")
    return float(np.quantile(state[valid, 0] / max_depth[valid], 0.90))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-fraction", type=float, default=0.20)
    ap.add_argument("--max-fit-groups", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = V60TrainCache(args.cache_manifest)
    graph = _load_graph(Path(args.graph))
    prepared = prepare_static_v60(graph, "cpu")
    all_d2 = cache.names("D2")
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=all_d2, holdout_fraction=args.holdout_fraction
    )
    if args.max_fit_groups > 0:
        fit = sorted(fit)[: args.max_fit_groups]
    if not fit:
        raise RuntimeError("V112 has no TrainFit D2 groups")

    # Thresholds use TrainFit outcomes only. Holdout labels are never batched/read.
    scales = derive_effect_scales_v111(cache, fit, prepared)
    horizon = HydraulicHorizonV110()
    idx = np.asarray(horizon.indices(), dtype=np.int64)
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    loads = {name: _load(cache.entry(name), max_depth) for name in fit}
    q1, q2 = np.quantile(np.asarray(list(loads.values())), [1/3, 2/3]).tolist()
    phase = lambda x: "low" if x <= q1 else ("mid" if x <= q2 else "high")

    def accumulator():
        return InfluencePriorAccumulatorV112(
            len(graph.actuator_ids), len(idx), len(graph.node_ids)
        )
    overall = accumulator()
    phased = {key: accumulator() for key in ("low", "mid", "high")}
    valid_probes = 0

    for n, name in enumerate(fit, 1):
        entry = cache.entry(name); arrays = entry.arrays; ref = entry.reference_index
        rs = np.asarray(arrays["settings"][ref], dtype=np.float64)
        rstate = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        rflow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        for cand in (i for i in entry.indices if i != ref):
            cs = np.asarray(arrays["settings"][cand], dtype=np.float64)
            source = infer_single_changed_actuator_v112(rs, cs)
            target = build_influence_support_targets_v112(
                reference_states=rstate,
                candidate_states=np.asarray(arrays["target_states"][cand], dtype=np.float64),
                reference_flows=rflow,
                candidate_flows=np.asarray(arrays["target_actuator_flows"][cand], dtype=np.float64),
                state_active_threshold=scales.state_active_threshold,
                flow_active_threshold=scales.flow_active_threshold,
                retained_indices=idx,
            )
            overall.update(source, target); phased[phase(loads[name])].update(source, target)
            valid_probes += 1
        if n == 1 or n % 10 == 0 or n == len(fit):
            print(f"[V112_ATLAS] groups={n}/{len(fit)} probes={valid_probes}", flush=True)

    prior = overall.finalize()
    arrays = {f"overall__{k}": v for k, v in prior.items() if k != "contract"}
    for key, acc in phased.items():
        arrays.update({f"{key}__{k}": v for k, v in acc.finalize().items() if k != "contract"})
    np.savez_compressed(out / "STEP2_V112_INFLUENCE_PRIOR.npz", **arrays)

    cache_branches = _branches(cache, all_d2)
    exposure = prior["exposure"]
    summary = {
        "contract": CONTRACT,
        "git_head": _head(),
        "development_only": True,
        "new_swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "internal_holdout_outcomes_accessed": False,
        "d2_source_lineage": classify_d2_population_v112(
            authoritative_branches=cache_branches,
            checkpoint_states=len(all_d2),
            actuator_count=len(graph.actuator_ids),
            event_count=_unique(cache, all_d2, "event_id"),
            group_count=len(all_d2),
        ),
        "derived_cache": {
            "groups": len(all_d2), "branches_including_reference": cache_branches,
            "events": _unique(cache, all_d2, "event_id"),
        },
        "trainfit": {
            "groups": len(fit), "branches_including_reference": _branches(cache, fit),
            "candidate_probes": valid_probes, "events": _unique(cache, fit, "event_id"),
        },
        "internal_holdout_metadata_only": {
            "groups": len(holdout), "events": _unique(cache, holdout, "event_id"),
        },
        "state_conditioning": {
            "metric": "causal_checkpoint_p90_depth_over_max_depth",
            "trainfit_tertile_boundaries": [q1, q2], "phases": ["low", "mid", "high"],
        },
        "meaningful_effect_threshold_contract": scales.contract,
        "retained_indices": idx.tolist(),
        "response_minutes": list(horizon.response_minutes()),
        "actuator_coverage": {
            "observed": int(np.sum(exposure > 0)), "total": len(graph.actuator_ids),
            "min_exposure": int(exposure.min()), "all_observed": bool(np.all(exposure > 0)),
        },
        "scientific_contract": {
            "d2_role": "soft state-conditioned support prior",
            "d3_role": "direct authoritative joint-action signed response",
            "sum_d2_magnitudes_for_d3": False,
            "hard_hop_cutoff": False,
            "zero_action_effect_exact_zero_required": True,
        },
    }
    (out / "STEP2_V112_INFLUENCE_ATLAS.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
