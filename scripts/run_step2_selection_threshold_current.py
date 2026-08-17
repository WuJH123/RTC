"""Calibrate the HOLD/action admission threshold for a frozen selection-aware Direct-TFV model.

This runner does not retrain Step2 and does not launch SWMM.  It reuses the rainfall-disjoint D3
HOLD-reference calibration/audit split, but replaces the previous max-residual scalar margin with an
empirical minimum-predicted-improvement threshold aligned to the actual online decision.

D4 AUDIT remains untouched by fitting and is used only as a distribution-shift stress test.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time

import torch

from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_tfv_selection_v2 import (
    DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT,
    calibrate_minimum_predicted_improvement,
    evaluate_selection_threshold,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60

from run_step2_selection_calibration_current import (
    EXPECTED_COUNTS,
    _collect_group_records,
    _d3_calibration_audit_split,
    _load_model,
    _normalization_from_checkpoint,
    _sha,
)


CURRENT_SELECTION_THRESHOLD_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_SELECTION_THRESHOLD_V2"
REPORT_FILENAME = "STEP2_DIRECT_TFV_SELECTION_THRESHOLD_REPORT.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    return p


def _evaluate(records: dict[str, object], threshold_m3: float) -> dict[str, float | int]:
    return evaluate_selection_threshold(
        best_candidate_prediction_m3=records["best_candidate_prediction_m3"],
        best_candidate_truth_m3=records["best_candidate_truth_m3"],
        oracle_truth_m3=records["oracle_truth_m3"],
        oracle_is_hold=records["oracle_is_hold"],
        minimum_predicted_improvement_m3=float(threshold_m3),
    )


def main() -> None:
    started = time.perf_counter()
    args = _parser().parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "action_support" not in checkpoint:
        raise ValueError("selection-threshold runner requires a selection-aware DEV checkpoint with action_support")
    if int(checkpoint["action_support"].get("single_facility_coverage_count", -1)) != 109:
        raise ValueError("selection-threshold runner requires 109/109 exact single-facility TrainFit coverage")
    model = _load_model(checkpoint, device)
    normalization = _normalization_from_checkpoint(checkpoint)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("selection threshold requires exactly 109 actuators")

    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate()
    state_store.validate()

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    counts = tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3, d4_fit_names, d4_audit_names)))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"current Development split differs from {EXPECTED_COUNTS}: got {counts}")

    d3_calib, d3_audit, d3_calib_rain, d3_audit_rain = _d3_calibration_audit_split(hold_d3)
    if set(d3_calib_rain) & set(d3_audit_rain):
        raise RuntimeError("selection calibration/audit rainfall groups overlap")
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    d4_fit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain_store), state_store)
    d4_audit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain_store), state_store)

    records = {
        "d3_selection_calibration": _collect_group_records(
            model, cache=base_online, names=d3_calib, normalization=normalization, graph=graph, device=device
        ),
        "d3_selection_audit": _collect_group_records(
            model, cache=base_online, names=d3_audit, normalization=normalization, graph=graph, device=device
        ),
        "d2_holdout_stress": _collect_group_records(
            model, cache=base_online, names=hold_d2, normalization=normalization, graph=graph, device=device
        ),
        "d4_fit_stress": _collect_group_records(
            model, cache=d4_fit_online, names=d4_fit_names, normalization=normalization, graph=graph, device=device
        ),
        "d4_audit_stress": _collect_group_records(
            model, cache=d4_audit_online, names=d4_audit_names, normalization=normalization, graph=graph, device=device
        ),
    }
    calibration_records = records["d3_selection_calibration"]
    calibration = calibrate_minimum_predicted_improvement(
        best_candidate_prediction_m3=calibration_records["best_candidate_prediction_m3"],
        best_candidate_truth_m3=calibration_records["best_candidate_truth_m3"],
        oracle_truth_m3=calibration_records["oracle_truth_m3"],
        oracle_is_hold=calibration_records["oracle_is_hold"],
    )
    threshold = float(calibration.minimum_predicted_improvement_m3)
    evaluations = {
        label: {
            "raw": _evaluate(value, 0.0),
            "thresholded": _evaluate(value, threshold),
        }
        for label, value in records.items()
    }

    report = {
        "contract": CURRENT_SELECTION_THRESHOLD_RUN_CONTRACT,
        "selection_contract": DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT,
        "development_only": True,
        "model_retrained": False,
        "swmm_launched": False,
        "d4_audit_used_for_calibration": False,
        "calibration_source": "rainfall-group-separated subset of existing D3 HOLD-reference internal holdout",
        "calibration_reference_family": "HOLD",
        "future_step3_reference_family": "HOLD",
        "calibration": asdict(calibration),
        "d3_split": {
            "calibration_group_count": len(d3_calib),
            "audit_group_count": len(d3_audit),
            "calibration_rainfall_groups": d3_calib_rain,
            "audit_rainfall_groups": d3_audit_rain,
            "rainfall_overlap": sorted(set(d3_calib_rain) & set(d3_audit_rain)),
        },
        "evaluations": evaluations,
        "action_support_contract": str(checkpoint["action_support"].get("contract")),
        "single_facility_training_coverage_count": int(
            checkpoint["action_support"].get("single_facility_coverage_count", -1)
        ),
        "lineage": {
            "checkpoint_sha256": _sha(args.checkpoint),
            "graph_sha256": _sha(args.graph),
            "base_cache_sha256": _sha(args.cache_manifest),
            "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
            "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "resources": {
            "device": str(device),
            "wall_seconds": float(time.perf_counter() - started),
        },
        "runtime_promoted": False,
        "next_gate": "trust-region Step3 solver-only Development audit using the calibrated minimum predicted improvement",
    }
    (out / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
