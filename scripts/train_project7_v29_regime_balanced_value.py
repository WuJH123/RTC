"""Train Project7 V29 regime-balanced value correction from existing q95 truth.

The script reuses the existing V28 augmented exact-return dataset. It does not
run SWMM. Q27 stays frozen, Train leakage groups fit/select the continuous
regime correction, and Validation/Test are report-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_project7_v28_q95_residual_value as v28train
from rtc.direct_tfv_operational_v27_runtime import build_operational_v27_controller
from rtc.direct_tfv_operational_v28_runtime import _q27_m3_from_latent, _v23_strong_storm_blend
from rtc.direct_tfv_v28_residual_value import build_v28_residual_features
from rtc.direct_tfv_v29_regime_value import (
    V29_FEATURE_NAMES,
    V29_REGIME_VALUE_CONTRACT,
    build_v29_regime_features,
    checkpoint_payload,
    evaluate_v29_value,
    fit_v29_regime_value,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path

V29_TRAINING_CONTRACT = "PROJECT7_STEP3_V29_REGIME_BALANCED_TRAIN_GROUP_CV_V1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_report(
    model: Any,
    features: np.ndarray,
    q27: np.ndarray,
    truth: np.ndarray,
    units: list[str],
    groups: list[str],
    stress: np.ndarray,
    *,
    stress_edges: tuple[float, float, float],
) -> dict[str, Any]:
    report = evaluate_v29_value(model, features, q27, truth, units, groups)
    q25, q50, q75 = stress_edges
    bins = (
        ("low", -np.inf, q25),
        ("low_mid", q25, q50),
        ("high_mid", q50, q75),
        ("high", q75, np.inf),
    )
    regime: dict[str, Any] = {}
    for name, lower, upper in bins:
        mask = (stress >= lower) & (stress < upper)
        if not mask.any():
            regime[name] = {"record_count": 0}
            continue
        idx = np.flatnonzero(mask)
        regime[name] = {
            "record_count": int(mask.sum()),
            "stress_min": float(np.min(stress[mask])),
            "stress_max": float(np.max(stress[mask])),
            "metrics": evaluate_v29_value(
                model,
                features[mask],
                q27[mask],
                truth[mask],
                [units[i] for i in idx],
                [groups[i] for i in idx],
            ),
        }
    report.update(
        {
            "record_count": int(len(truth)),
            "decision_unit_count": int(len(set(units))),
            "leakage_group_count": int(len(set(groups))),
            "stress_regime_metrics": regime,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument("--v27-dataset-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-records", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--study-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V29 training requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    manifest_path = Path(args.dataset_manifest).resolve()
    records_path = Path(args.dataset_records).resolve()
    v27_dataset_path = Path(args.v27_dataset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = v28train._rows(records_path)
    if _sha(records_path).lower() != str(manifest.get("records_sha256", "")).lower():
        raise ValueError("V29 dataset records SHA mismatch")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise ValueError("V29 dataset record count mismatch")
    leakage = manifest.get("leakage_audit")
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise ValueError("V29 dataset leakage audit must pass")
    if set(str(row.get("split", "")) for row in rows) != {"train", "validation", "test"}:
        raise ValueError("V29 dataset must contain Train/Validation/Test")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    step2_path = practical_asset_path(assets, "step2")
    parent, _, _, parent_lineage = build_operational_v27_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=step2_path,
        supervisory_control_path=control_path,
        sequence_support_path=support_path,
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        v27_value_checkpoint_path=args.v27_value_checkpoint,
        dataset_manifest_path=v27_dataset_path,
        asset_manifest_path=args.asset_manifest,
        device=device,
        decision_runtime_budget_seconds=180.0,
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    mpc = parent.controller._direct_mpc_adapter.inner
    study_root = Path(args.study_root).resolve() if args.study_root else None

    feature_rows: list[np.ndarray] = []
    q27_rows: list[float] = []
    truth_rows: list[float] = []
    split_rows: list[str] = []
    units: list[str] = []
    groups: list[str] = []
    sources: list[str] = []
    stress_rows: list[float] = []
    rejected: dict[str, int] = {}

    for row in rows:
        try:
            context_path = v28train._recover_context_path(row, study_root=study_root)
            expected_context_sha = str(row.get("context_npz_sha256", "")).lower()
            if len(expected_context_sha) == 64 and _sha(context_path).lower() != expected_context_sha:
                raise ValueError("context_sha_mismatch")
            ctx = v28train._context(context_path, device=device)
            stored = torch.as_tensor(
                row["candidate_target"], dtype=torch.float32, device=device
            ).reshape(-1)
            supported, _, changed, support = mpc._h10_supported_target(stored, ctx["active"])
            if changed <= 0:
                raise ValueError("q95_projection_is_hold")
            if not torch.equal(stored.to(torch.float32), supported.to(torch.float32)):
                raise ValueError("stored_action_not_exact_q95_supported")
            geometry = v28train._geometry(
                mpc,
                raw_target=stored,
                supported_target=supported,
                active_target=ctx["active"],
            )
            with torch.inference_mode():
                built, latent_tensor, _, _ = mpc._feature_and_scores(
                    current_state=ctx["state"],
                    rainfall=ctx["rain"],
                    flow=ctx["flow"].reshape(1, -1),
                    active_target=ctx["active"],
                    target=supported,
                    source=str(row.get("candidate_source", "HISTORICAL_EXACT_ACTION")),
                )
            q27 = _q27_m3_from_latent(mpc.v27_value_model, float(latent_tensor))
            v28_feature = build_v28_residual_features(
                q27_score_m3=q27,
                q95_scale=geometry["q95_scale"],
                q95_max_ratio=float(support["max_ratio"]),
                q95_binding=bool(support["binding"]),
                raw_first_move_l1=geometry["raw_first_move_l1"],
                supported_first_move_l1=geometry["supported_first_move_l1"],
                raw_to_supported_first_move_l1=geometry["raw_to_supported_first_move_l1"],
                raw_to_supported_h120_l1=geometry["raw_to_supported_h120_l1"],
                raw_to_supported_tv_l1=geometry["raw_to_supported_tv_l1"],
                changed_facility_count=int(changed),
                network_stress_q75=float(built.base.network_stress_q75),
                rain_level=float(built.base.rain_level),
                strong_storm_blend=_v23_strong_storm_blend(
                    built.base.network_stress_q75
                ),
                candidate_source=str(row.get("candidate_source", "HISTORICAL_EXACT_ACTION")),
            )
            feature_rows.append(build_v29_regime_features(v28_feature))
            q27_rows.append(float(q27))
            truth_rows.append(float(row["true_policy_return_delta_tfv_m3"]))
            split_rows.append(str(row["split"]))
            units.append(v28train._decision_unit(row))
            groups.append(str(row["leakage_group_id"]))
            sources.append(str(row.get("candidate_source", "")))
            stress_rows.append(float(built.base.network_stress_q75))
        except FileNotFoundError:
            rejected["missing_context"] = rejected.get("missing_context", 0) + 1
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            text = str(exc)
            reason = (
                text
                if text
                in {
                    "context_sha_mismatch",
                    "q95_projection_is_hold",
                    "stored_action_not_exact_q95_supported",
                }
                else type(exc).__name__
            )
            rejected[reason] = rejected.get(reason, 0) + 1

    if not feature_rows:
        raise RuntimeError("V29 has no exact q95-supported rows")
    x = np.stack(feature_rows).astype(np.float64)
    q27 = np.asarray(q27_rows, dtype=np.float64)
    truth = np.asarray(truth_rows, dtype=np.float64)
    stress = np.asarray(stress_rows, dtype=np.float64)
    split = np.asarray(split_rows, dtype=object)
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    if not train.any() or not validation.any() or not test.any():
        raise RuntimeError("V29 q95-compatible rows left an empty split")

    train_idx = np.flatnonzero(train)
    model, selection = fit_v29_regime_value(
        train_features=x[train],
        train_q27_scores_m3=q27[train],
        train_truth_m3=truth[train],
        train_groups=[groups[i] for i in train_idx],
        train_units=[units[i] for i in train_idx],
        q27_checkpoint_sha256=_sha(args.v27_value_checkpoint),
        cv_folds=int(args.cv_folds),
        seed=int(args.seed),
    )
    stress_edges = (
        float(selection["train_stress_q25"]),
        float(selection["train_stress_q50"]),
        float(selection["train_stress_q75"]),
    )
    split_reports: dict[str, Any] = {}
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        idx = np.flatnonzero(mask)
        split_reports[name] = _split_report(
            model,
            x[mask],
            q27[mask],
            truth[mask],
            [units[i] for i in idx],
            [groups[i] for i in idx],
            stress[mask],
            stress_edges=stress_edges,
        )

    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V29 output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    control_payload = json.loads(Path(control_path).read_text(encoding="utf-8"))
    lineage = {
        "base_step2_sha256": _sha(step2_path),
        "q27_checkpoint_sha256": _sha(args.v27_value_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "dataset_manifest_sha256": _sha(manifest_path),
        "v27_parent_dataset_manifest_sha256": _sha(v27_dataset_path),
        "dataset_records_sha256": _sha(records_path),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(support_path),
        "v15_rank_checkpoint_sha256_parent_compatibility_only": _sha(
            args.v15_rank_checkpoint
        ),
        "v21_boundary_checkpoint_sha256_parent_compatibility_only": _sha(
            args.v21_boundary_checkpoint
        ),
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "feature_contract": "|".join(V29_FEATURE_NAMES),
        "v27_parent_lineage": dict(parent_lineage),
        "q95_exact_supported_rows_only": True,
        "group_balanced_train_weighting": True,
        "new_swmm_truth_generated": False,
        "return_period_used_as_feature": False,
        "event_duration_used_as_feature": False,
        "event_id_used_as_feature": False,
    }
    payload = checkpoint_payload(
        model,
        lineage=lineage,
        selection_report=selection,
        validation_report=split_reports["validation"],
        test_report=split_reports["test"],
    )
    checkpoint_path = out_dir / "V29_REGIME_BALANCED_VALUE_MODEL.pt"
    torch.save(payload, checkpoint_path)
    report = {
        "contract": V29_TRAINING_CONTRACT,
        "model_contract": V29_REGIME_VALUE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "record_count": int(len(truth)),
        "input_record_count": int(len(rows)),
        "q95_exact_supported_record_count": int(len(truth)),
        "rejected_counts": dict(sorted(rejected.items())),
        "feature_width": int(x.shape[1]),
        "feature_names": list(V29_FEATURE_NAMES),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "selection": selection,
        "selected_ridge": float(model.ridge),
        "selected_shrinkage": float(model.shrinkage),
        "split_reports": split_reports,
        "validation_used_for_model_selection": False,
        "test_used_for_training_or_model_selection": False,
        "return_period_used_as_feature": False,
        "event_duration_used_as_feature": False,
        "lineage": lineage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "development_only": True,
        "formal_evidence": False,
        "ready_for_policy_lock": False,
    }
    report_path = out_dir / "V29_REGIME_BALANCED_VALUE_MODEL_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
