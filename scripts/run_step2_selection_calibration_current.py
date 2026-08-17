"""Calibrate a Development-only HOLD/action guard for a frozen Direct-TFV V2 checkpoint.

This runner does not retrain Step2 and does not launch SWMM. It takes the already trained full-DEV
pairwise value checkpoint, splits the existing D3 HOLD-reference internal holdout by rainfall group
into calibration/audit subsets, calibrates a one-sided selected-action residual margin on the D3
calibration subset only, and then reports raw versus guarded selection on D3 audit, D2 holdout,
D4 FIT and untouched D4 AUDIT.

D4 AUDIT never enters margin fitting. The output is diagnostic evidence for deciding whether the
next step can be a Development Step3 solver test or whether new HOLD-reference first-move evidence
is actually necessary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_tfv_selection import (
    DIRECT_TFV_SELECTION_CONTRACT,
    DirectTFVSelectionDesign,
    calibrate_selected_action_margin,
    evaluate_selection_margin,
)
from rtc.step2_tfv_value import DIRECT_TFV_VALUE_CONTRACT, DirectFacilityTFVValueModel, DirectTFVValueDesign
from rtc.step2_tfv_value_training import _branch_indices, _forward_candidates, _graph_tensors
from rtc.step2_train_response_v60 import InputNormalizationV60, V60TrainCache, deterministic_rainfall_split_v60


CURRENT_SELECTION_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_SELECTION_CALIBRATION_V1"
REPORT_FILENAME = "STEP2_DIRECT_TFV_SELECTION_REPORT.json"
SEED = 42
EXPECTED_COUNTS = (112, 112, 32, 32, 33, 15)


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    p.add_argument("--alpha", type=float, default=0.10)
    return p


def _rainfall_group(name: str) -> str:
    parts = str(name).split("::")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"cannot parse rainfall group from {name!r}")
    return parts[1]


def _d3_calibration_audit_split(names: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    rainfall_groups = sorted({_rainfall_group(name) for name in names})
    if len(rainfall_groups) < 2:
        raise ValueError("D3 selection calibration requires at least two held-out rainfall groups")
    ordered = sorted(
        rainfall_groups,
        key=lambda value: hashlib.sha256(f"selection|{SEED}|{value}".encode("utf-8")).hexdigest(),
    )
    cut = max(1, min(len(ordered) - 1, len(ordered) // 2))
    calibration_groups = set(ordered[:cut])
    calibration = sorted(name for name in names if _rainfall_group(name) in calibration_groups)
    audit = sorted(name for name in names if _rainfall_group(name) not in calibration_groups)
    if not calibration or not audit:
        raise ValueError("D3 rainfall-group calibration/audit split is empty")
    return calibration, audit, ordered[:cut], ordered[cut:]


def _normalization_from_checkpoint(checkpoint: dict) -> InputNormalizationV60:
    values = checkpoint["normalization"]
    return InputNormalizationV60(
        state_mean=np.asarray(values["state_mean"], dtype=np.float32),
        state_std=np.asarray(values["state_std"], dtype=np.float32),
        rainfall_mean=np.asarray(values["rainfall_mean"], dtype=np.float32),
        rainfall_std=np.asarray(values["rainfall_std"], dtype=np.float32),
        flow_mean=np.asarray(values["flow_mean"], dtype=np.float32),
        flow_std=np.asarray(values["flow_std"], dtype=np.float32),
    )


def _load_model(checkpoint: dict, device: torch.device) -> DirectFacilityTFVValueModel:
    if str(checkpoint.get("contract")) != DIRECT_TFV_VALUE_CONTRACT:
        raise ValueError("selection calibration requires the current Direct-TFV V2 checkpoint")
    if str(checkpoint.get("profile")) != "dev":
        raise ValueError("selection calibration requires a full DEV checkpoint, not smoke")
    counts = checkpoint.get("selected_group_counts", {})
    observed = tuple(int(counts.get(key, -1)) for key in (
        "fit_d2", "fit_d3", "hold_d2", "hold_d3", "d4_fit", "d4_audit"
    ))
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"full DEV checkpoint counts differ from {EXPECTED_COUNTS}: got {observed}")
    design = DirectTFVValueDesign(**dict(checkpoint["model_design"]))
    model = DirectFacilityTFVValueModel(
        state_dim=int(checkpoint["state_dim"]),
        rainfall_dim=int(checkpoint["rainfall_dim"]),
        actuator_physics_dim=int(checkpoint["actuator_physics_dim"]),
        target_scale_m3=float(checkpoint["target_scale_m3"]),
        design=design,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _collect_group_records(
    model: DirectFacilityTFVValueModel,
    *,
    cache,
    names: list[str],
    normalization: InputNormalizationV60,
    graph,
    device: torch.device,
) -> dict[str, object]:
    static = _graph_tensors(graph, device)
    group_names: list[str] = []
    best_prediction: list[float] = []
    best_truth: list[float] = []
    oracle_truth: list[float] = []
    oracle_is_hold: list[bool] = []
    candidate_counts: list[int] = []
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            indices = _branch_indices(batch, mode="all")
            if int(indices.numel()) <= 0:
                continue
            output = _forward_candidates(model, batch, indices, graph_tensors=static)
            prediction = output.total_delta_tfv_m3.detach().cpu().numpy().astype(np.float64)
            truth = (
                batch.true_delta_tfv_m3[0]
                .index_select(0, indices)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            best = int(np.argmin(prediction))
            truth_with_hold = np.concatenate((np.zeros(1, dtype=np.float64), truth))
            oracle = int(np.argmin(truth_with_hold))
            group_names.append(str(name))
            best_prediction.append(float(prediction[best]))
            best_truth.append(float(truth[best]))
            oracle_truth.append(float(truth_with_hold[oracle]))
            oracle_is_hold.append(bool(oracle == 0))
            candidate_counts.append(int(len(truth)))
    if not group_names:
        raise ValueError("selection calibration found no scorable groups")
    return {
        "names": group_names,
        "best_candidate_prediction_m3": best_prediction,
        "best_candidate_truth_m3": best_truth,
        "oracle_truth_m3": oracle_truth,
        "oracle_is_hold": oracle_is_hold,
        "candidate_counts": candidate_counts,
    }


def _evaluate(records: dict[str, object], *, margin_m3: float) -> dict[str, float | int]:
    return evaluate_selection_margin(
        best_candidate_prediction_m3=records["best_candidate_prediction_m3"],
        best_candidate_truth_m3=records["best_candidate_truth_m3"],
        oracle_truth_m3=records["oracle_truth_m3"],
        oracle_is_hold=records["oracle_is_hold"],
        margin_m3=margin_m3,
    )


def main() -> None:
    started = time.perf_counter()
    args = _parser().parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint, device)
    normalization = _normalization_from_checkpoint(checkpoint)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("Direct-TFV selection calibration requires exactly 109 actuators")

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
    selection_design = DirectTFVSelectionDesign(alpha=float(args.alpha))
    calibration = calibrate_selected_action_margin(
        best_candidate_prediction_m3=records["d3_selection_calibration"]["best_candidate_prediction_m3"],
        best_candidate_truth_m3=records["d3_selection_calibration"]["best_candidate_truth_m3"],
        design=selection_design,
    )
    evaluations = {
        label: {
            "raw": _evaluate(value, margin_m3=0.0),
            "guarded": _evaluate(value, margin_m3=float(calibration.margin_m3)),
        }
        for label, value in records.items()
    }

    report = {
        "contract": CURRENT_SELECTION_RUN_CONTRACT,
        "selection_contract": DIRECT_TFV_SELECTION_CONTRACT,
        "development_only": True,
        "model_retrained": False,
        "swmm_launched": False,
        "d4_audit_used_for_calibration": False,
        "calibration_source": "rainfall-group-separated subset of existing D3 internal holdout",
        "calibration_reference_family": "HOLD",
        "future_step3_reference_family": "HOLD",
        "distribution_shift_warning": "D4 stress tests use Sparse-RBC references, so D3-calibrated margin is diagnostic there and not a formal coverage claim",
        "selection_design": asdict(selection_design),
        "calibration": asdict(calibration),
        "d3_split": {
            "calibration_group_count": len(d3_calib),
            "audit_group_count": len(d3_audit),
            "calibration_rainfall_groups": d3_calib_rain,
            "audit_rainfall_groups": d3_audit_rain,
        },
        "evaluations": evaluations,
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
        "next_gate": "decide whether a conservative HOLD/action guard improves selected true delta TFV without collapsing action rate before any Step3 closed-loop wiring",
    }
    (out / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
