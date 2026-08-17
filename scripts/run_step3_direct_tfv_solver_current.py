"""Run the current Direct-TFV Step3 solver on untouched HOLD-reference Development states.

This is a solver-only audit: it loads the selection-aware full-DEV checkpoint and the independently
calibrated HOLD/action threshold, screens all 109 facilities, optimises a TrainFit-supported active
set, and saves the proposed H360 sequences.  It does not launch SWMM and therefore makes no claim of
authoritative control benefit.  Passing this gate is the prerequisite for a small authoritative
SWMM first-move/closed-loop probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_tfv_selection_v2 import DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT
from rtc.step2_tfv_value_training import _branch_indices, _forward_candidates, _graph_tensors
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step3_tfv_value_mpc_v2 import DirectTFVMPCDesignV2, DirectTFVTrustRegionMPC

from run_step2_selection_calibration_current import (
    EXPECTED_COUNTS,
    _d3_calibration_audit_split,
    _load_model,
    _normalization_from_checkpoint,
)


CURRENT_STEP3_SOLVER_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_STEP3_SOLVER_AUDIT_V1"
REPORT_FILENAME = "STEP3_DIRECT_TFV_SOLVER_REPORT.json"
SEQUENCES_FILENAME = "STEP3_DIRECT_TFV_PROPOSED_SEQUENCES.npz"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--selection-report", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-groups", type=int, default=0)
    return p


def _physical_inputs(batch, normalization):
    state_mean = torch.as_tensor(normalization.state_mean, dtype=batch.initial_state.dtype, device=batch.initial_state.device)
    state_std = torch.as_tensor(normalization.state_std, dtype=batch.initial_state.dtype, device=batch.initial_state.device)
    rain_mean = torch.as_tensor(normalization.rainfall_mean, dtype=batch.rainfall.dtype, device=batch.rainfall.device)
    rain_std = torch.as_tensor(normalization.rainfall_std, dtype=batch.rainfall.dtype, device=batch.rainfall.device)
    flow_mean = torch.as_tensor(normalization.flow_mean, dtype=batch.previous_actuator_flow.dtype, device=batch.previous_actuator_flow.device)
    flow_std = torch.as_tensor(normalization.flow_std, dtype=batch.previous_actuator_flow.dtype, device=batch.previous_actuator_flow.device)
    return (
        batch.initial_state * state_std + state_mean,
        batch.rainfall * rain_std + rain_mean,
        batch.previous_actuator_flow * flow_std + flow_mean,
    )


def main() -> None:
    started = time.perf_counter()
    args = _parser().parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "action_support" not in checkpoint:
        raise ValueError("Step3 solver requires a checkpoint with TrainFit action_support")
    if int(checkpoint["action_support"].get("single_facility_coverage_count", -1)) != 109:
        raise ValueError("Step3 solver requires 109/109 single-facility TrainFit coverage")
    model = _load_model(checkpoint, device)
    normalization = _normalization_from_checkpoint(checkpoint)
    graph = _load_graph(args.graph)
    if tuple(checkpoint["action_support"].get("actuator_ids", ())) != tuple(graph.actuator_ids):
        raise ValueError("checkpoint action-support actuator order differs from graph")

    selection = json.loads(Path(args.selection_report).read_text(encoding="utf-8"))
    if str(selection.get("selection_contract")) != DIRECT_TFV_SELECTION_THRESHOLD_CONTRACT:
        raise ValueError("Step3 solver requires the current HOLD/action threshold report")
    if str(selection.get("lineage", {}).get("checkpoint_sha256", "")).lower() != _sha(args.checkpoint).lower():
        raise ValueError("selection threshold was calibrated on a different Step2 checkpoint")
    if selection.get("d3_split", {}).get("rainfall_overlap"):
        raise ValueError("selection calibration/audit rainfall groups overlap")
    threshold = float(selection["calibration"]["minimum_predicted_improvement_m3"])

    base = V60TrainCache(args.cache_manifest)
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
    # D4 counts are validated by the full checkpoint/selection report.  The solver audit itself uses
    # only HOLD-reference D3 audit states because that matches future Step3 reference semantics.
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != EXPECTED_COUNTS[:4]:
        raise ValueError("base D2/D3 Development split changed")
    _, d3_audit, _, d3_audit_rain = _d3_calibration_audit_split(hold_d3)
    expected_audit_rain = list(selection["d3_split"]["audit_rainfall_groups"])
    if d3_audit_rain != expected_audit_rain:
        raise ValueError("Step3 D3 audit rainfall groups differ from selection report")
    if int(args.max_groups) > 0:
        d3_audit = d3_audit[: int(args.max_groups)]
    if not d3_audit:
        raise ValueError("Step3 solver audit has no D3 audit groups")

    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    design = DirectTFVMPCDesignV2(minimum_predicted_improvement_m3=threshold)
    controller = DirectTFVTrustRegionMPC(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        design=design,
    )
    static = _graph_tensors(graph, device)
    records: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    names: list[str] = []
    support_violations = engineering_violations = 0
    action_count = 0
    training_max = float(checkpoint["action_support"].get("absolute_delta_tfv_max_m3", 0.0))
    extreme_prediction_count = 0

    for name in d3_audit:
        batch = online.batch(name, normalization, device)
        reference = batch.reference_settings[0]
        active_target = reference[0]
        if float(torch.max(torch.abs(reference - active_target[None])).detach().cpu()) > 1.0e-6:
            raise ValueError(f"{name}: D3 audit reference is not HOLD across H72")
        state_raw, rain_raw, flow_raw = _physical_inputs(batch, normalization)
        result = controller.optimize(
            current_state=state_raw,
            rainfall=rain_raw,
            previous_actuator_flow=flow_raw,
            current_settings=active_target,
            active_target=active_target,
        )
        indices = _branch_indices(batch, mode="all")
        cached = _forward_candidates(model, batch, indices, graph_tensors=static)
        cached_prediction = cached.total_delta_tfv_m3.detach().cpu().numpy().astype(np.float64)
        cached_truth = (
            batch.true_delta_tfv_m3[0]
            .index_select(0, indices)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        best_cached = int(np.argmin(cached_prediction))
        first = result.settings[: design.control_block_steps].mean(dim=0)
        physical_lo = torch.as_tensor(controller.min_setting, dtype=first.dtype, device=first.device)
        physical_hi = torch.as_tensor(controller.max_setting, dtype=first.dtype, device=first.device)
        physical_bad = bool(torch.any(first < physical_lo - 1.0e-6) or torch.any(first > physical_hi + 1.0e-6))
        support_bad = float(result.maximum_support_ratio) > 1.0001
        engineering_violations += int(physical_bad or result.first_move_changed_facility_count > result.active_facility_count)
        support_violations += int(support_bad)
        action_count += int(result.selected_source == "DIRECT_TFV_TRUST_REGION_LBFGSB")
        if training_max > 0.0 and abs(float(result.predicted_delta_tfv_m3)) > 1.25 * training_max:
            extreme_prediction_count += 1
        records.append(
            {
                "group": name,
                "selected_source": result.selected_source,
                "predicted_delta_tfv_m3": float(result.predicted_delta_tfv_m3),
                "best_cached_prediction_m3": float(cached_prediction[best_cached]),
                "best_cached_true_delta_tfv_m3": float(cached_truth[best_cached]),
                "predicted_gain_beyond_best_cached_m3": float(
                    cached_prediction[best_cached] - float(result.predicted_delta_tfv_m3)
                ),
                "screened_facility_count": int(result.screened_facility_count),
                "active_facility_count": int(result.active_facility_count),
                "active_facility_ids": list(result.active_facility_ids),
                "active_facility_screening_scores_m3": list(result.active_facility_screening_scores_m3),
                "first_move_changed_facility_count": int(result.first_move_changed_facility_count),
                "maximum_support_ratio": float(result.maximum_support_ratio),
                "optimizer_success": bool(result.optimizer_success),
                "optimizer_steps": int(result.optimizer_steps),
                "optimizer_starts": int(result.optimizer_starts),
                "gradient_norm": float(result.gradient_norm),
                "solver_elapsed_seconds": float(result.elapsed_seconds),
                "scipy_message": result.scipy_message,
                "physical_or_active_set_violation": bool(
                    physical_bad or result.first_move_changed_facility_count > result.active_facility_count
                ),
                "support_violation": bool(support_bad),
            }
        )
        names.append(name)
        sequences.append(result.settings.detach().cpu().numpy().astype(np.float32))

    elapsed = float(time.perf_counter() - started)
    action_fraction = float(action_count / len(records))
    solver_times = np.asarray([float(record["solver_elapsed_seconds"]) for record in records], dtype=np.float64)
    ready = bool(
        support_violations == 0
        and engineering_violations == 0
        and action_count > 0
        and np.isfinite(solver_times).all()
        and float(np.max(solver_times)) < float(design.deadline_seconds)
    )
    np.savez_compressed(
        out / SEQUENCES_FILENAME,
        group_name=np.asarray(names),
        settings=np.stack(sequences),
        actuator_ids=np.asarray(graph.actuator_ids),
    )
    report = {
        "contract": CURRENT_STEP3_SOLVER_RUN_CONTRACT,
        "development_only": True,
        "swmm_launched": False,
        "step2_model_retrained": False,
        "selection_threshold_recalibrated": False,
        "minimum_predicted_improvement_m3": threshold,
        "screen_all_109_facilities": True,
        "trust_region_from_trainfit_action_support": True,
        "single_facility_training_coverage_count": int(
            checkpoint["action_support"]["single_facility_coverage_count"]
        ),
        "training_joint_changed_facility_q50": float(
            checkpoint["action_support"]["joint_changed_facility_count_q50"]
        ),
        "groups": len(records),
        "action_selected_count": int(action_count),
        "action_selected_fraction": action_fraction,
        "support_violation_count": int(support_violations),
        "engineering_violation_count": int(engineering_violations),
        "extreme_prediction_vs_trainfit_max_count": int(extreme_prediction_count),
        "solver_elapsed_seconds_median": float(np.median(solver_times)),
        "solver_elapsed_seconds_max": float(np.max(solver_times)),
        "ready_for_authoritative_swmm_probe": ready,
        "records": records,
        "lineage": {
            "checkpoint_sha256": _sha(args.checkpoint),
            "selection_report_sha256": _sha(args.selection_report),
            "graph_sha256": _sha(args.graph),
            "base_cache_sha256": _sha(args.cache_manifest),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "resources": {"device": str(device), "wall_seconds": elapsed},
        "next_gate": "small authoritative SWMM replay of proposed first moves before any multi-event closed loop",
    }
    (out / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
