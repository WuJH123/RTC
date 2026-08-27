"""Train Project7 V27 decision-aware exact-return value model.

Train leakage groups are used for internal CV; a shortlist is finalized on Validation.  Test is read
only after model selection.  Pointwise exact-return and same-state pairwise ranking supervision are
both extracted from the same leakage-safe historical dataset.
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
from rtc.direct_tfv_policy_return_portfolio_v23 import V23_HYDRAULIC_CANDIDATE_CONTRACT, V23_PORTFOLIO_CONTRACT
from rtc.direct_tfv_v26_value_model import candidate_metrics, decision_metrics
from rtc.direct_tfv_v27_decision_value import (
    V27_VALUE_MODEL_CONTRACT,
    checkpoint_payload,
    fit_v27_decision_value_model,
    pairwise_rank_accuracy,
)
from rtc.direct_tfv_v27_hydraulic_features import (
    V27_HYDRAULIC_FEATURE_CONTRACT,
    build_v27_hydraulic_feature,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


V27_TRAINING_CONTRACT = "PROJECT7_STEP3_V27_DECISION_AWARE_TRAIN_VALIDATION_TEST_V1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"invalid V27 records: {path}")
    return values


def _context(path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=False)
    try:
        needed = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
        if any(key not in data for key in needed):
            raise ValueError(f"V27 context lacks causal fields: {path}")
        state = torch.as_tensor(np.asarray(data["current_state"]), dtype=torch.float32, device=device)
        rain = torch.as_tensor(np.asarray(data["rainfall_scenarios"]), dtype=torch.float32, device=device)
        active = torch.as_tensor(np.asarray(data["active_target"]), dtype=torch.float32, device=device).reshape(-1)
        flow = torch.as_tensor(np.asarray(data["previous_actuator_flow"]), dtype=torch.float32, device=device).reshape(-1)
    finally:
        data.close()
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if rain.ndim == 5 and int(rain.shape[0]) == 1:
        rain = rain[0]
    if state.ndim != 2 or rain.ndim != 4 or tuple(active.shape) != (109,) or tuple(flow.shape) != (109,):
        raise ValueError(f"invalid V27 context shape: {path}")
    return {"state": state, "rain": rain, "active": active, "flow": flow}


def _decision_unit(row: dict[str, Any]) -> str:
    context = str(row.get("causal_context_fingerprint_sha256", "")).strip().lower()
    group = str(row.get("leakage_group_id", "")).strip().lower()
    if not context or not group:
        raise ValueError("V27 decision unit requires leakage group and causal context")
    return hashlib.sha256(f"{group}|{context}".encode("utf-8")).hexdigest()


def _split_report(model, x, y, units) -> dict[str, Any]:
    latent = model.latent_numpy(x)
    prediction = model.predict_m3_numpy(x)
    return {
        "candidate_metrics": candidate_metrics(prediction, y),
        "decision_metrics": decision_metrics(latent, y, units),
        "pairwise_metrics": pairwise_rank_accuracy(latent, y, units),
        "latent_min": float(latent.min()),
        "latent_max": float(latent.max()),
        "latent_std": float(latent.std()),
        "report_clip_hit_count": int(model.clip_hit_numpy(x).sum()),
        "report_clip_hit_fraction": float(model.clip_hit_numpy(x).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-records", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V27 training requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    manifest_path = Path(args.dataset_manifest).resolve()
    records_path = Path(args.dataset_records).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _rows(records_path)
    if str(manifest.get("records_sha256", "")).lower() != _sha(records_path).lower():
        raise ValueError("V27 dataset records SHA mismatch")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise ValueError("V27 dataset record count mismatch")
    leakage = manifest.get("leakage_audit")
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise ValueError("V27 dataset must carry a passed leakage audit")
    if set(str(row.get("split", "")) for row in rows) != {"train", "validation", "test"}:
        raise ValueError("V27 dataset must contain Train/Validation/Test")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    controller, _, _, parent_lineage = build_operational_v23_controller(
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
    splits: list[str] = []
    decision_units: list[str] = []
    leakage_groups: list[str] = []
    sources: list[str] = []
    stress: list[float] = []
    for row in rows:
        context_path = Path(str(row["context_npz"])).resolve()
        if not context_path.is_file():
            raise FileNotFoundError(context_path)
        expected_sha = str(row.get("context_npz_sha256", "")).lower()
        if len(expected_sha) == 64 and _sha(context_path).lower() != expected_sha:
            raise ValueError(f"V27 context SHA mismatch: {row.get('query_set_id', '')}")
        ctx = _context(context_path, device=device)
        target = torch.as_tensor(row["candidate_target"], dtype=torch.float32, device=device).reshape(-1)
        if tuple(target.shape) != (109,):
            raise ValueError("V27 candidate target must be [109]")
        with torch.inference_mode():
            built = build_v27_hydraulic_feature(
                step2_model=mpc.model,
                normalization=mpc.normalization,
                graph=mpc.graph,
                current_state=ctx["state"],
                rainfall_scenarios=ctx["rain"],
                previous_actuator_flow=ctx["flow"],
                active_target=ctx["active"],
                candidate_target=target,
                candidate_source=str(row.get("candidate_source", "HISTORICAL_EXACT_ACTION")),
                supervisory_mask=mpc.supervisory_mask,
                target_scale_m3=float(mpc.model.target_scale_m3.detach().cpu()),
            )
        features.append(built.feature.detach().cpu().numpy().astype(np.float64))
        truth.append(float(row["true_policy_return_delta_tfv_m3"]))
        splits.append(str(row["split"]))
        decision_units.append(_decision_unit(row))
        leakage_groups.append(str(row["leakage_group_id"]))
        sources.append(str(row.get("candidate_source", "")))
        stress.append(float(built.base.network_stress_q75))

    x = np.stack(features)
    y = np.asarray(truth, dtype=np.float64)
    split_array = np.asarray(splits, dtype=object)
    train = split_array == "train"
    validation = split_array == "validation"
    test = split_array == "test"

    model, selection = fit_v27_decision_value_model(
        x[train],
        y[train],
        [decision_units[i] for i in np.flatnonzero(train)],
        [leakage_groups[i] for i in np.flatnonzero(train)],
        x[validation],
        y[validation],
        [decision_units[i] for i in np.flatnonzero(validation)],
        seed=int(args.seed),
        cv_folds=int(args.cv_folds),
    )
    split_reports: dict[str, Any] = {}
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        units = [decision_units[i] for i in np.flatnonzero(mask)]
        groups = {leakage_groups[i] for i in np.flatnonzero(mask)}
        report = _split_report(model, x[mask], y[mask], units)
        report.update(
            {
                "record_count": int(mask.sum()),
                "decision_unit_count": len(set(units)),
                "leakage_group_count": len(groups),
            }
        )
        split_reports[name] = report

    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    control_payload = json.loads(Path(control_path).read_text(encoding="utf-8"))
    lineage = {
        "base_step2_sha256": _sha(step2_path),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(support_path),
        "dataset_manifest_sha256": _sha(manifest_path),
        "dataset_records_sha256": _sha(records_path),
        "feature_contract": V27_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "v23_hydraulic_candidate_contract": V23_HYDRAULIC_CANDIDATE_CONTRACT,
        "decision_unit_contract": "SHA256_LEAKAGE_GROUP_CANONICAL_CAUSAL_CONTEXT",
        "v15_rank_checkpoint_sha256_parent_compatibility_only": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256_parent_compatibility_only": _sha(args.v21_boundary_checkpoint),
        "v15_rank_used_for_v27_candidate_selection": False,
        "v21_boundary_used_for_v27_action_admission": False,
    }

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V27 training output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model,
        lineage=lineage,
        training_report=selection,
        validation_report=split_reports["validation"],
        test_report=split_reports["test"],
    )
    checkpoint_path = out_dir / "V27_DECISION_AWARE_EXACT_RETURN_VALUE_MODEL.pt"
    torch.save(payload, checkpoint_path)

    stress_array = np.asarray(stress, dtype=np.float64)
    report = {
        "contract": V27_TRAINING_CONTRACT,
        "model_contract": V27_VALUE_MODEL_CONTRACT,
        "feature_contract": V27_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "record_count": len(rows),
        "feature_width": int(x.shape[1]),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "split_reports": split_reports,
        "model_selection": selection,
        "selected_ridge": float(model.ridge),
        "selected_pairwise_weight": float(model.pairwise_weight),
        "context_support": {
            "stress_min": float(stress_array.min()),
            "stress_median": float(np.median(stress_array)),
            "stress_max": float(stress_array.max()),
            "stress_std": float(stress_array.std()),
        },
        "lineage": lineage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "runtime_ranking_uses_unclipped_latent": True,
        "reporting_inverse_transform_can_clip_without_changing_rank": True,
        "test_used_for_training_or_model_selection": False,
        "scientific_metrics_block_runtime": False,
        "development_only": True,
        "formal_evidence": False,
        "ready_for_policy_lock": False,
    }
    report_path = out_dir / "V27_DECISION_AWARE_EXACT_RETURN_VALUE_MODEL_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
