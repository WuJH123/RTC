"""Fit V25R2 on the exact current-policy TFV return already present in the V25 matched bank.

This is a target-contract correction, not a new data-generation step.  It reuses the same 48
rainfall-group-disjoint matched contexts and candidate branches, but supervises the value gate with
``true_policy_return_delta_tfv_m3`` (candidate H10 followed by the frozen continuation versus HOLD
followed by the identical continuation).  H30/H60/H120 fields remain diagnostics only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.direct_tfv_v25_features import build_v25_selected_query
from rtc.direct_tfv_v25r2_value_calibration import (
    V25R2_ESTIMAND,
    V25R2_TRUTH_FIELD,
    V25R2_VALUE_CALIBRATOR_CONTRACT,
    V25R2_VALUE_FEATURE_CONTRACT,
    fit_v25_value_calibrator,
    fitted_payload_v25r2,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or invalid JSONL: {path}")
    return rows


def _context(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    needed = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
    if any(key not in data for key in needed):
        raise ValueError(f"context is missing required causal fields: {path}")
    return {key: np.asarray(data[key]).copy() for key in needed}


def _torch_context(row: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    context = _context(Path(str(row["context_npz"])).resolve())
    state = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
    rain = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
    flow = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
    active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
    if state.ndim == 2:
        state = state[None]
    if rain.ndim == 5 and int(rain.shape[0]) == 1:
        rain = rain[0]
    if flow.ndim == 1:
        flow = flow[None]
    if active.ndim == 2 and int(active.shape[0]) == 1:
        active = active[0]
    return {"current_state": state, "rainfall": rain, "flow": flow, "active": active}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--matched-bank-manifest", required=True)
    parser.add_argument("--matched-records", required=True)
    parser.add_argument("--context-records", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V25R2 training requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    bank_manifest_path = Path(args.matched_bank_manifest).resolve()
    records_path = Path(args.matched_records).resolve()
    context_records_path = Path(args.context_records).resolve()
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    bank_rows = _records(records_path)
    context_rows = _records(context_records_path)
    if int(bank_manifest.get("record_count", -1)) != len(bank_rows) or len(bank_rows) != 48:
        raise ValueError("V25R2 matched bank must contain exactly 48 selected-query records")
    if str(bank_manifest.get("records_sha256", "")).lower() != _sha(records_path).lower():
        raise ValueError("V25R2 matched bank records SHA mismatch")
    context_by_query = {
        str(row["query_set_id"]): row
        for row in context_rows
        if str(row.get("candidate_source", "")) == "TYPE_AWARE_HYDRAULIC_PRESSURE"
    }
    if len(context_by_query) != 48:
        raise ValueError("V25R2 context records must contain 48 hydraulic Train contexts")
    if len({str(row["rainfall_group"]) for row in bank_rows}) != 48:
        raise ValueError("V25R2 matched bank rainfall groups are not unique")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    controller, _graph, _, v23_lineage = build_operational_v23_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=step2_path,
        supervisory_control_path=control_path,
        sequence_support_path=support_path,
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        device=device,
        decision_runtime_budget_seconds=180.0,
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    mpc = controller.controller._direct_mpc_adapter.inner
    features: list[np.ndarray] = []
    truth: list[float] = []
    groups: list[str] = []
    stresses: list[float] = []
    blends: list[float] = []
    selected_sources: dict[str, int] = {}

    for bank_row in sorted(bank_rows, key=lambda row: str(row["query_set_id"])):
        query_id = str(bank_row["query_set_id"])
        context_row = context_by_query.get(query_id)
        if context_row is None:
            raise ValueError(f"missing causal context for matched query {query_id}")
        if V25R2_TRUTH_FIELD not in bank_row:
            raise ValueError(f"matched record lacks exact policy return for {query_id}")
        branch_delta = float(bank_row["candidate_branch_tfv_m3"]) - float(bank_row["hold_branch_tfv_m3"])
        exact_delta = float(bank_row[V25R2_TRUTH_FIELD])
        if abs(branch_delta - exact_delta) > 1.0e-6:
            raise ValueError(f"exact policy-return arithmetic mismatch for {query_id}")
        if bank_row.get("same_prefix_verified") is not True or bank_row.get("same_continuation_policy_verified") is not True:
            raise ValueError(f"matched policy-return lineage is incomplete for {query_id}")
        context = _torch_context(context_row, device=device)
        with torch.inference_mode():
            selected = build_v25_selected_query(
                mpc=mpc,
                current_state=context["current_state"],
                rainfall=context["rainfall"],
                previous_actuator_flow=context["flow"],
                active_target=context["active"],
            )
        expected_hash = str(bank_row["candidate_first_target_sha256"]).lower()
        if selected.selected_target_sha256.lower() != expected_hash:
            raise ValueError(f"V25R2 selected action hash mismatch for {query_id}")
        if str(context_row.get("context_npz_sha256", "")).lower() != _sha(context_row["context_npz"]).lower():
            raise ValueError(f"context SHA mismatch for {query_id}")
        features.append(selected.feature.detach().cpu().numpy().astype(np.float64))
        truth.append(exact_delta)
        groups.append(str(bank_row["rainfall_group"]))
        stresses.append(float(selected.network_stress_q75))
        blends.append(float(selected.strong_storm_blend))
        selected_sources[selected.selected_source] = selected_sources.get(selected.selected_source, 0) + 1

    x = np.stack(features)
    y = np.asarray(truth, dtype=np.float64)
    fitted, oof_metrics, oof_prediction = fit_v25_value_calibrator(x, y, groups)
    stress = np.asarray(stresses, dtype=np.float64)
    blend = np.asarray(blends, dtype=np.float64)
    context_support = {
        "count": int(len(stress)),
        "network_stress_min": float(stress.min()),
        "network_stress_median": float(np.median(stress)),
        "network_stress_max": float(stress.max()),
        "network_stress_std": float(stress.std()),
        "strong_storm_blend_positive_count": int(np.sum(blend > 0.0)),
        "strong_storm_blend_ge_0_4_count": int(np.sum(blend >= 0.4)),
        "production_stress_support_warning": bool(not np.any(blend > 0.0)),
        "role": "diagnostic_only_not_a_threshold_tuning_signal",
    }
    control_payload = json.loads(Path(control_path).read_text(encoding="utf-8"))
    lineage = {
        "base_step2_sha256": _sha(step2_path),
        "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(support_path),
        "matched_truth_bank_manifest_sha256": _sha(bank_manifest_path),
        "matched_truth_records_sha256": _sha(records_path),
        "v23_portfolio_contract": str(v23_lineage["v23_portfolio_contract"]),
        "v23_hydraulic_candidate_contract": str(v23_lineage["v23_hydraulic_candidate_contract"]),
        "feature_contract": V25R2_VALUE_FEATURE_CONTRACT,
        "data_role": "policy_return_train",
        "truth_field": V25R2_TRUTH_FIELD,
        "estimand": V25R2_ESTIMAND,
        "boundary_supervision_unit": "CURRENT_V23_RANK_SELECTED_CANDIDATE_PER_QUERY",
        "group_disjoint_oof": True,
        "new_rainfall_generated": False,
        "new_validation_truth_used": False,
        "new_calibration_truth_used": False,
        "additional_swmm_runs_for_v25r2_training": 0,
    }
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V25R2 calibrator output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = fitted_payload_v25r2(fitted, lineage=lineage, oof_metrics=oof_metrics)
    checkpoint_path = out_dir / "V25R2_EXACT_POLICY_RETURN_CALIBRATOR.pt"
    torch.save(payload, checkpoint_path)
    audit_rows = [
        {
            "query_set_id": str(bank_rows[index]["query_set_id"]),
            "rainfall_group": groups[index],
            "truth_exact_delta_tfv_m3": float(y[index]),
            "oof_predicted_delta_tfv_m3": float(oof_prediction[index]),
            "oof_upper_bound_m3": float(oof_prediction[index] + fitted.one_sided_error_margin_m3),
            "decision": "ACTION" if float(oof_prediction[index] + fitted.one_sided_error_margin_m3) < 0.0 else "HOLD",
            "network_stress_q75": float(stress[index]),
            "strong_storm_blend": float(blend[index]),
        }
        for index in range(len(y))
    ]
    report = {
        "contract": V25R2_VALUE_CALIBRATOR_CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "target_contract_correction_from_v25_v1": True,
        "v25_v1_h120_target_is_diagnostic_only": True,
        "estimand": V25R2_ESTIMAND,
        "truth_field": V25R2_TRUTH_FIELD,
        "matched_record_count": len(bank_rows),
        "selected_sources": selected_sources,
        "feature_contract": V25R2_VALUE_FEATURE_CONTRACT,
        "feature_width": int(x.shape[1]),
        "oof_metrics": oof_metrics,
        "context_support": context_support,
        "lineage": lineage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "additional_swmm_runs": 0,
        "train_oof_supported": bool(oof_metrics.get("train_oof_supported", False)),
        "offline_gate": {
            "supported": bool(oof_metrics.get("train_oof_supported", False)),
            "reason": "PASS" if bool(oof_metrics.get("train_oof_supported", False)) else "EXACT_RETURN_OOF_VALUE_ADMISSION_NOT_SAFE",
        },
        "oof_rows": audit_rows,
    }
    report_path = out_dir / "V25R2_EXACT_POLICY_RETURN_CALIBRATOR_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "train_oof_supported": report["train_oof_supported"],
        "oof_metrics": oof_metrics,
        "context_support": context_support,
        "additional_swmm_runs": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
