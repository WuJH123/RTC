"""Write a decision-level, no-SWMM audit of the frozen V25 matched Train OOF gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.direct_tfv_v25_features import build_v25_selected_query
from rtc.direct_tfv_v25_value_calibration import fit_v25_value_calibrator
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path

from train_project7_v25_value_calibrator_current import _records, _torch_context, _sha


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
        raise RuntimeError("V25 audit requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    bank_rows = _records(Path(args.matched_records).resolve())
    context_rows = _records(Path(args.context_records).resolve())
    context_by_query = {
        str(row["query_set_id"]): row
        for row in context_rows
        if str(row.get("candidate_source", "")) == "TYPE_AWARE_HYDRAULIC_PRESSURE"
    }
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    controller, _, _, _ = build_operational_v23_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        supervisory_control_path=practical_asset_path(assets, "supervisory_control"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
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
    rows: list[dict[str, Any]] = []
    for bank_row in sorted(bank_rows, key=lambda row: str(row["query_set_id"])):
        query_id = str(bank_row["query_set_id"])
        context_row = context_by_query[query_id]
        context = _torch_context(context_row, device=device)
        with torch.inference_mode():
            selected = build_v25_selected_query(
                mpc=mpc,
                current_state=context["current_state"],
                rainfall=context["rainfall"],
                previous_actuator_flow=context["flow"],
                active_target=context["active"],
            )
        if selected.selected_target_sha256.lower() != str(bank_row["candidate_first_target_sha256"]).lower():
            raise ValueError(f"selected candidate hash mismatch: {query_id}")
        features.append(selected.feature.detach().cpu().numpy().astype(np.float64))
        truth.append(float(bank_row["true_policy_return_delta_tfv_h120_m3"]))
        groups.append(str(bank_row["rainfall_group"]))
        rows.append(
            {
                "query_set_id": query_id,
                "event_id": str(bank_row["event_id"]),
                "rainfall_group": str(bank_row["rainfall_group"]),
                "candidate_source": selected.selected_source,
                "candidate_first_target_sha256": selected.selected_target_sha256,
                "actual_delta_tfv_h120_m3": float(bank_row["true_policy_return_delta_tfv_h120_m3"]),
                "beneficial_actual": bool(float(bank_row["true_policy_return_delta_tfv_h120_m3"]) < 0.0),
                "network_stress_q75": selected.network_stress_q75,
                "strong_storm_blend": selected.strong_storm_blend,
                "candidate_count": selected.candidate_count,
                "selected_changed_facility_count": selected.selected_changed_facility_count,
            }
        )
    fitted, metrics, oof = fit_v25_value_calibrator(
        np.stack(features), np.asarray(truth, dtype=np.float64), groups
    )
    margin = float(fitted.one_sided_error_margin_m3)
    for row, prediction in zip(rows, oof):
        upper = float(prediction + margin)
        action = bool(upper < 0.0)
        row.update(
            {
                "oof_predicted_delta_tfv_h120_m3": float(prediction),
                "tfv_value_upper_bound_m3": upper,
                "decision": "ACTION" if action else "HOLD",
                "tfv_value_gate_reason": "TFV_UCB_NEGATIVE" if action else "TFV_UCB_NONNEGATIVE",
            }
        )
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"V25 audit output is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "PROJECT7_STEP3_V25_MATCHED_VALUE_GATE_DECISION_AUDIT_V1",
        "development_only": True,
        "formal_evidence": False,
        "new_swmm_runs": 0,
        "matched_bank_manifest": str(Path(args.matched_bank_manifest).resolve()),
        "matched_bank_manifest_sha256": _sha(args.matched_bank_manifest),
        "matched_records": str(Path(args.matched_records).resolve()),
        "matched_records_sha256": _sha(args.matched_records),
        "feature_count": len(rows),
        "feature_width": int(np.stack(features).shape[1]),
        "one_sided_error_margin_m3": margin,
        "oof_metrics": metrics,
        "rows": rows,
    }
    json_path = out / "V25_VALUE_GATE_DECISION_AUDIT.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = out / "V25_VALUE_GATE_DECISION_AUDIT.csv"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out / "V25_VALUE_GATE_DECISION_AUDIT.md"
    lines = [
        "# Project7 V25 matched value-gate audit",
        "",
        "Development-only; no Validation/Final truth and no SWMM runs in this audit.",
        "",
        f"- OOF AUC (beneficial vs harmful): {metrics['auc_beneficial_vs_harmful']:.6f}",
        f"- OOF margin: {margin:.6f} m3",
        f"- OOF ACTION: {metrics['action_count']}/{metrics['count']}",
        f"- Harmful admitted ACTION: {metrics['harmful_action_count']}/{metrics['action_count']}",
        f"- train_oof_supported: {metrics['train_oof_supported']}",
        "",
        "| event | query | candidate | predicted m3 | UCB m3 | decision | actual delta m3 |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['event_id']} | {row['query_set_id'][:12]} | {row['candidate_source']} | "
            f"{row['oof_predicted_delta_tfv_h120_m3']:.3f} | {row['tfv_value_upper_bound_m3']:.3f} | "
            f"{row['decision']} | {row['actual_delta_tfv_h120_m3']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path), "oof_metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
