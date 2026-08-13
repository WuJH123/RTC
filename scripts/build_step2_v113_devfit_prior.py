"""Build the frozen V113 support prior from the nested DevFit events only.

This is deliberately separate from the historical V112 atlas.  The nested
DevCheck events are read only to identify groups; their hydraulic outcomes are
never loaded while constructing the prior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_influence_prior_v112 import InfluencePriorAccumulatorV112
from rtc.step2_influence_support_v112 import (
    build_influence_support_targets_v112,
    infer_single_changed_actuator_v112,
)
from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step2_v110_contract import HydraulicHorizonV110
from run_step2_v110 import _load_graph


CONTRACT = "PROJECT7_STEP2_V113_DEVFIT_SOFT_SUPPORT_PRIOR_V1"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nested-split", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = V60TrainCache(args.cache_manifest)
    graph = _load_graph(args.graph)
    split = json.loads(Path(args.nested_split).read_text(encoding="utf-8"))
    fit_names = [str(x) for x in split["v113_devfit"]["group_names"]]
    check_names = [str(x) for x in split["v113_devcheck"]["group_names"]]
    if not fit_names or not check_names:
        raise ValueError("nested V113 split must contain non-empty DevFit and DevCheck groups")
    fit_events = {cache.entry(n).event_id for n in fit_names}
    check_events = {cache.entry(n).event_id for n in check_names}
    if fit_events & check_events:
        raise ValueError("nested V113 prior has event leakage")
    if not all(n.startswith("D2::") for n in fit_names + check_names):
        raise ValueError("V113 nested prior requires D2 groups only")

    prepared = prepare_static_v60(graph, "cpu")
    scales = derive_effect_scales_v111(cache, fit_names, prepared)
    horizon = HydraulicHorizonV110()
    retained = np.asarray(horizon.indices(), dtype=np.int64)
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    phase_load: dict[str, float] = {}
    for name in fit_names:
        state = np.asarray(cache.entry(name).arrays["initial_state"][cache.entry(name).reference_index], dtype=np.float64)
        valid = (max_depth > 1e-6) & np.isfinite(state[:, 0])
        phase_load[name] = float(np.quantile(state[valid, 0] / max_depth[valid], 0.90))
    q1, q2 = np.quantile(np.asarray(list(phase_load.values())), [1.0 / 3.0, 2.0 / 3.0]).tolist()

    def phase_of(value: float) -> str:
        return "low" if value <= q1 else ("mid" if value <= q2 else "high")

    def new_acc() -> InfluencePriorAccumulatorV112:
        return InfluencePriorAccumulatorV112(len(graph.actuator_ids), len(retained), len(graph.node_ids))

    overall = new_acc()
    phased = {name: new_acc() for name in ("low", "mid", "high")}
    probes = 0
    # Only these arrays are authoritative DevFit outcomes.  DevCheck names are
    # intentionally not touched below; they remain a held-out development check.
    for name in fit_names:
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        ref_settings = np.asarray(arrays["settings"][ref], dtype=np.float64)
        ref_states = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        ref_flows = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        phase = phase_of(phase_load[name])
        for cand in (i for i in entry.indices if i != ref):
            source = infer_single_changed_actuator_v112(ref_settings, np.asarray(arrays["settings"][cand], dtype=np.float64))
            target = build_influence_support_targets_v112(
                reference_states=ref_states,
                candidate_states=np.asarray(arrays["target_states"][cand], dtype=np.float64),
                reference_flows=ref_flows,
                candidate_flows=np.asarray(arrays["target_actuator_flows"][cand], dtype=np.float64),
                state_active_threshold=scales.state_active_threshold,
                flow_active_threshold=scales.flow_active_threshold,
                retained_indices=retained,
            )
            overall.update(source, target)
            phased[phase].update(source, target)
            probes += 1

    prior = overall.finalize()
    arrays = {f"overall__{k}": v for k, v in prior.items() if k != "contract"}
    for phase, acc in phased.items():
        arrays.update({f"{phase}__{k}": v for k, v in acc.finalize().items() if k != "contract"})
    prior_path = out / "STEP2_V113_DEVFIT_SUPPORT_PRIOR.npz"
    np.savez_compressed(prior_path, **arrays)
    report = {
        "contract": CONTRACT,
        "git_head": _head(),
        "development_only": True,
        "new_swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "devcheck_outcomes_accessed": False,
        "nested_split_sha256": _sha(args.nested_split),
        "cache_manifest_sha256": _sha(args.cache_manifest),
        "graph_sha256": _sha(args.graph),
        "prior_sha256": _sha(prior_path),
        "source_events": sorted(fit_events),
        "source_groups": len(fit_names),
        "source_candidate_probes": probes,
        "held_out_events": sorted(check_events),
        "held_out_groups": len(check_names),
        "retained_indices": retained.tolist(),
        "phase_boundaries": {"q1_p90_depth_fill": float(q1), "q2_p90_depth_fill": float(q2)},
        "phase_group_counts": {p: int(sum(phase_of(v) == p for v in phase_load.values())) for p in ("low", "mid", "high")},
        "active_threshold_contract": scales.contract,
        "soft_prior_only": True,
        "hard_support_mask": False,
        "sum_signed_d2_magnitudes": False,
    }
    (out / "STEP2_V113_DEVFIT_SUPPORT_PRIOR.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"prior": str(prior_path), "groups": len(fit_names), "probes": probes, "events": len(fit_events)}, indent=2))


if __name__ == "__main__":
    main()
