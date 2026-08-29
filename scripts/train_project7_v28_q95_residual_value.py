"""Train the Project7 V28 q95-matched residual value correction.

This Development-only trainer reuses the V27 exact-return dataset and frozen V27 model.  Rows are
admitted only when their stored candidate target is already exactly the q95-supported target in the
current frozen sequence contract; no new SWMM truth is generated here.  Train groups fit the ridge,
Validation selects it, and Test is read once for reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v27_runtime import build_operational_v27_controller
from rtc.direct_tfv_operational_v28_runtime import _q27_m3_from_latent, _v23_strong_storm_blend
from rtc.direct_tfv_policy_return import encode_policy_return_action_token
from rtc.direct_tfv_v28_residual_value import (
    V28_RESIDUAL_FEATURE_NAMES,
    V28_RESIDUAL_MODEL_CONTRACT,
    checkpoint_payload,
    build_v28_residual_features,
    evaluate_v28_residual,
    fit_v28_residual_value_model,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


V28_TRAINING_CONTRACT = "PROJECT7_STEP3_V28_Q95_MATCHED_RESIDUAL_VALUE_TRAIN_VALIDATION_TEST_V1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"invalid V28 dataset records: {path}")
    return values


def _context(path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=False)
    try:
        needed = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
        if any(key not in data for key in needed):
            raise ValueError(f"V28 context lacks causal fields: {path}")
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
        raise ValueError(f"invalid V28 context shape: {path}")
    return {"state": state, "rain": rain, "active": active, "flow": flow}


def _decision_unit(row: dict[str, Any]) -> str:
    group = str(row.get("leakage_group_id", "")).strip().lower()
    context = str(row.get("causal_context_fingerprint_sha256", "")).strip().lower()
    if not group or not context:
        raise ValueError("V28 decision unit requires leakage group and causal context")
    return hashlib.sha256(f"{group}|{context}".encode("utf-8")).hexdigest()


def _geometry(
    mpc: Any,
    *,
    raw_target: torch.Tensor,
    supported_target: torch.Tensor,
    active_target: torch.Tensor,
) -> dict[str, Any]:
    _, raw_batch = encode_policy_return_action_token(
        active_target,
        raw_target,
        horizon_steps=int(mpc.design.prediction_horizon_steps),
        first_action_steps=int(mpc.design.control_block_steps),
    )
    _, supported_batch = encode_policy_return_action_token(
        active_target,
        supported_target,
        horizon_steps=int(mpc.design.prediction_horizon_steps),
        first_action_steps=int(mpc.design.control_block_steps),
    )
    raw_sequence = raw_batch[0].detach()
    supported_sequence = supported_batch[0].detach()
    supported_sequence = mpc._contract_to_joint_sequence_support(supported_sequence, active_target).detach()
    raw_geometry = mpc._joint_sequence_geometry_torch(raw_sequence, active_target)
    supported_geometry = mpc._joint_sequence_geometry_torch(supported_sequence, active_target)
    raw_first = float(raw_geometry["first_block_l1"].detach().cpu())
    supported_first = float(supported_geometry["first_block_l1"].detach().cpu())
    raw_delta = raw_sequence[1:] - raw_sequence[:-1]
    supported_delta = supported_sequence[1:] - supported_sequence[:-1]
    return {
        "raw_sequence": raw_sequence,
        "supported_sequence": supported_sequence,
        "raw_first_move_l1": raw_first,
        "supported_first_move_l1": supported_first,
        "q95_scale": float(np.clip(supported_first / raw_first if raw_first > 1.0e-12 else 1.0, 0.0, 1.0)),
        "raw_to_supported_first_move_l1": float(torch.abs(raw_target - supported_target).sum().detach().cpu()),
        "raw_to_supported_h120_l1": float(torch.abs(raw_sequence - supported_sequence).sum().detach().cpu()),
        "raw_to_supported_tv_l1": float(torch.abs(raw_delta - supported_delta).sum().detach().cpu()),
    }


def _recover_context_path(row: dict[str, Any], *, study_root: Path | None) -> Path:
    requested = Path(str(row.get("context_npz", "")))
    if requested.is_file():
        return requested.resolve()
    if study_root is not None:
        matches = sorted(study_root.rglob(requested.name))
        expected = str(row.get("context_npz_sha256", "")).lower()
        for match in matches:
            if len(expected) != 64 or _sha(match).lower() == expected:
                return match.resolve()
    raise FileNotFoundError(requested)


def _split_report(
    model: Any,
    features: np.ndarray,
    q27: np.ndarray,
    truth: np.ndarray,
    units: list[str],
    groups: list[str],
) -> dict[str, Any]:
    report = evaluate_v28_residual(model, features, q27, truth, units)
    report.update(
        {
            "record_count": int(len(truth)),
            "decision_unit_count": int(len(set(units))),
            "leakage_group_count": int(len(set(groups))),
            "context_count": int(len(set(units))),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument(
        "--v27-dataset-manifest",
        help="Immutable V27 dataset manifest used only to validate/load the frozen Q27 checkpoint",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-records", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--study-root")
    parser.add_argument("--truth-plan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V28 training requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    manifest_path = Path(args.dataset_manifest).resolve()
    v27_dataset_manifest_path = (
        Path(args.v27_dataset_manifest).resolve() if args.v27_dataset_manifest else manifest_path
    )
    records_path = Path(args.dataset_records).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _rows(records_path)
    if _sha(records_path).lower() != str(manifest.get("records_sha256", "")).lower():
        raise ValueError("V28 dataset records SHA mismatch")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise ValueError("V28 dataset record count mismatch")
    leakage = manifest.get("leakage_audit")
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise ValueError("V28 dataset leakage audit must pass")
    if set(str(row.get("split", "")) for row in rows) != {"train", "validation", "test"}:
        raise ValueError("V28 dataset must contain Train/Validation/Test")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
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
        dataset_manifest_path=v27_dataset_manifest_path,
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
    stress_values: list[float] = []
    rain_values: list[float] = []
    rejected: dict[str, int] = {}
    for row in rows:
        try:
            context_path = _recover_context_path(row, study_root=study_root)
            expected_context_sha = str(row.get("context_npz_sha256", "")).lower()
            if len(expected_context_sha) == 64 and _sha(context_path).lower() != expected_context_sha:
                raise ValueError("context_sha_mismatch")
            ctx = _context(context_path, device=device)
            raw_target = torch.as_tensor(row["candidate_target"], dtype=torch.float32, device=device).reshape(-1)
            supported_target, _, changed, support = mpc._h10_supported_target(raw_target, ctx["active"])
            if changed <= 0:
                raise ValueError("q95_projection_is_hold")
            if not torch.equal(raw_target.to(torch.float32), supported_target.to(torch.float32)):
                raise ValueError("stored_action_not_exact_q95_supported")
            geometry = _geometry(
                mpc,
                raw_target=raw_target,
                supported_target=supported_target,
                active_target=ctx["active"],
            )
            with torch.inference_mode():
                built, latent_tensor, reported_tensor, clip_hit = mpc._feature_and_scores(
                    current_state=ctx["state"],
                    rainfall=ctx["rain"],
                    flow=ctx["flow"].reshape(1, -1),
                    active_target=ctx["active"],
                    target=supported_target,
                    source=str(row.get("candidate_source", "HISTORICAL_EXACT_ACTION")),
                )
            del reported_tensor, clip_hit
            latent = float(latent_tensor)
            q27 = _q27_m3_from_latent(mpc.v27_value_model, latent)
            feature = build_v28_residual_features(
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
                strong_storm_blend=_v23_strong_storm_blend(built.base.network_stress_q75),
                candidate_source=str(row.get("candidate_source", "HISTORICAL_EXACT_ACTION")),
            )
            feature_rows.append(feature)
            q27_rows.append(q27)
            truth_rows.append(float(row["true_policy_return_delta_tfv_m3"]))
            split_rows.append(str(row["split"]))
            units.append(_decision_unit(row))
            groups.append(str(row["leakage_group_id"]))
            sources.append(str(row.get("candidate_source", "")))
            stress_values.append(float(built.base.network_stress_q75))
            rain_values.append(float(built.base.rain_level))
        except FileNotFoundError:
            rejected["missing_context"] = rejected.get("missing_context", 0) + 1
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            reason = str(exc) if str(exc) in {"context_sha_mismatch", "q95_projection_is_hold", "stored_action_not_exact_q95_supported"} else type(exc).__name__
            rejected[reason] = rejected.get(reason, 0) + 1
    if not feature_rows:
        raise RuntimeError("V28 has no exact q95-supported rows")
    x = np.stack(feature_rows).astype(np.float64)
    q27 = np.asarray(q27_rows, dtype=np.float64)
    y = np.asarray(truth_rows, dtype=np.float64)
    split = np.asarray(split_rows, dtype=object)
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    if not train.any() or not validation.any() or not test.any():
        raise RuntimeError("q95-compatible V28 rows left an empty split")

    model, selection = fit_v28_residual_value_model(
        train_features=x[train],
        train_q27_scores_m3=q27[train],
        train_truth_m3=y[train],
        train_groups=[groups[i] for i in np.flatnonzero(train)],
        train_units=[units[i] for i in np.flatnonzero(train)],
        validation_features=x[validation],
        validation_q27_scores_m3=q27[validation],
        validation_truth_m3=y[validation],
        validation_units=[units[i] for i in np.flatnonzero(validation)],
        q27_checkpoint_sha256=_sha(args.v27_value_checkpoint),
    )
    split_reports: dict[str, Any] = {}
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        split_reports[name] = _split_report(
            model,
            x[mask],
            q27[mask],
            y[mask],
            [units[i] for i in np.flatnonzero(mask)],
            [groups[i] for i in np.flatnonzero(mask)],
        )
    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    control_payload = json.loads(Path(control_path).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V28 output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "base_step2_sha256": _sha(step2_path),
        "q27_checkpoint_sha256": _sha(args.v27_value_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "dataset_manifest_sha256": _sha(manifest_path),
        "v27_parent_dataset_manifest_sha256": _sha(v27_dataset_manifest_path),
        "dataset_records_sha256": _sha(records_path),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(support_path),
        "v15_rank_checkpoint_sha256_parent_compatibility_only": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256_parent_compatibility_only": _sha(args.v21_boundary_checkpoint),
        "v15_rank_used_for_v28_candidate_selection": False,
        "v21_boundary_used_for_v28_action_admission": False,
        "v25_ucb_used_for_v28_action_admission": False,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "feature_contract": "|".join(V28_RESIDUAL_FEATURE_NAMES),
        "v27_parent_lineage": dict(parent_lineage),
        "q95_exact_supported_rows_only": True,
        "new_swmm_truth_generated": False,
        "event_id_feature": False,
    }
    payload = checkpoint_payload(
        model,
        lineage=lineage,
        training_report=selection,
        validation_report=split_reports["validation"],
        test_report=split_reports["test"],
    )
    checkpoint_path = out_dir / "V28_Q95_MATCHED_RESIDUAL_VALUE_MODEL.pt"
    torch.save(payload, checkpoint_path)
    report = {
        "contract": V28_TRAINING_CONTRACT,
        "model_contract": V28_RESIDUAL_MODEL_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "record_count": int(len(y)),
        "input_record_count": int(len(rows)),
        "q95_exact_supported_record_count": int(len(y)),
        "rejected_counts": dict(sorted(rejected.items())),
        "feature_width": int(x.shape[1]),
        "feature_names": list(V28_RESIDUAL_FEATURE_NAMES),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "split_reports": split_reports,
        "validation_model_selection": selection,
        "selected_ridge": float(model.ridge),
        "pairwise_weight": 0.0,
        "context_support": {
            "stress_min": float(np.min(stress_values)),
            "stress_median": float(np.median(stress_values)),
            "stress_max": float(np.max(stress_values)),
            "stress_std": float(np.std(stress_values)),
            "rain_level_min": float(np.min(rain_values)),
            "rain_level_median": float(np.median(rain_values)),
            "rain_level_max": float(np.max(rain_values)),
        },
        "lineage": lineage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "test_used_for_training_or_model_selection": False,
        "scientific_metrics_block_runtime": False,
        "development_only": True,
        "formal_evidence": False,
        "ready_for_policy_lock": False,
    }
    report_path = out_dir / "V28_Q95_MATCHED_RESIDUAL_VALUE_MODEL_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
