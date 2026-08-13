"""TrainFit-only 0-360 min influence census; no new SWMM and no model-horizon change.

The current V11 Hydraulic learning window remains 0-120 min. This diagnostic
uses the original 360-min D2 outcomes only to quantify delayed/non-local effects
that would otherwise disappear from an influence census.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_influence_prior_v112 import InfluencePriorAccumulatorV112
from rtc.step2_influence_support_v112 import (
    STATE_EFFECT_NAMES_V112,
    build_influence_support_targets_v112,
    infer_single_changed_actuator_v112,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from run_step2_v110 import _load_graph

CONTRACT = "PROJECT7_STEP2_V112_TRAINFIT_INFLUENCE_LAG360_AUDIT_V1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cache = V60TrainCache(args.cache_manifest)
    graph = _load_graph(Path(args.graph))
    prepared = prepare_static_v60(graph, "cpu")
    fit, _ = deterministic_rainfall_split_v60(cache, names=cache.names("D2"), holdout_fraction=0.20)
    scales = derive_effect_scales_v111(cache, fit, prepared)  # threshold fitted on TrainFit only
    horizon = MultiResolutionHorizonV60()
    idx = np.asarray(horizon.indices(), dtype=np.int64)
    minutes = (idx + 1) * 5.0
    acc = InfluencePriorAccumulatorV112(len(graph.actuator_ids), len(idx), len(graph.node_ids))
    channel_active = np.zeros((len(idx), 5), dtype=np.int64)
    channel_total = np.zeros((len(idx), 5), dtype=np.int64)
    probes = 0

    for n, name in enumerate(fit, 1):
        entry = cache.entry(name); arrays = entry.arrays; ref = entry.reference_index
        ref_setting = np.asarray(arrays["settings"][ref], dtype=np.float64)
        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        for cand in (i for i in entry.indices if i != ref):
            source = infer_single_changed_actuator_v112(
                ref_setting, np.asarray(arrays["settings"][cand], dtype=np.float64)
            )
            target = build_influence_support_targets_v112(
                reference_states=ref_state,
                candidate_states=np.asarray(arrays["target_states"][cand], dtype=np.float64),
                reference_flows=ref_flow,
                candidate_flows=np.asarray(arrays["target_actuator_flows"][cand], dtype=np.float64),
                state_active_threshold=scales.state_active_threshold,
                flow_active_threshold=scales.flow_active_threshold,
                retained_indices=idx,
            )
            acc.update(source, target)
            channel_active += target.state_active.sum(axis=1).astype(np.int64)
            channel_total += np.asarray(target.state_active.shape[1], dtype=np.int64)
            probes += 1
        if n == 1 or n % 10 == 0 or n == len(fit):
            print(f"[V112_LAG360] groups={n}/{len(fit)} probes={probes}", flush=True)

    prior = acc.finalize()
    np.savez_compressed(
        out / "STEP2_V112_INFLUENCE_LAG360_PRIOR.npz",
        state_support_probability=prior["state_support_probability"],
        flow_support_probability=prior["flow_support_probability"],
        source_flow_effective_probability=prior["source_flow_effective_probability"],
        exposure=prior["exposure"], retained_indices=idx, response_minutes=minutes,
    )
    fractions = channel_active / np.maximum(channel_total, 1)
    buckets = {}
    for label, lo, hi in (("0_30", 0, 30), ("30_120", 30, 120), ("120_360", 120, 360)):
        mask = (minutes > lo) & (minutes <= hi)
        buckets[label] = {
            channel: float(fractions[mask, c].mean()) if mask.any() else 0.0
            for c, channel in enumerate(STATE_EFFECT_NAMES_V112)
        }
    summary = {
        "contract": CONTRACT,
        "development_only": True,
        "trainfit_groups": len(fit),
        "candidate_probes": probes,
        "new_swmm_run": False,
        "internal_holdout_outcomes_accessed": False,
        "validation_accessed": False,
        "final_accessed": False,
        "hydraulic_model_horizon_changed": False,
        "diagnostic_source_horizon_minutes": 360,
        "retained_indices": idx.tolist(),
        "response_minutes": minutes.tolist(),
        "active_cell_fraction_by_horizon_bucket": buckets,
        "interpretation": "120-360 min evidence is diagnostic for lag/nonlocal reach; V11 Hydraulic remains 0-120 min",
    }
    (out / "STEP2_V112_INFLUENCE_LAG360.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
