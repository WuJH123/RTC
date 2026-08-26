"""Train the V26 action-conditioned exact-return model on an explicit Train/Validation/Test bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.direct_tfv_v26_hydraulic_features import (
    V26_HYDRAULIC_FEATURE_CONTRACT,
    build_v26_hydraulic_feature,
)
from rtc.direct_tfv_v26_value_model import (
    V26_VALUE_MODEL_CONTRACT,
    candidate_metrics,
    checkpoint_payload,
    decision_metrics,
    fit_v26_value_model,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


V26_TRAINING_CONTRACT = "PROJECT7_STEP3_V26_TRAIN_VALIDATION_TEST_EXACT_RETURN_TRAINING_V1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or invalid V26 records: {path}")
    return rows


def _context(path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=False)
    needed = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
    if any(key not in data for key in needed):
        raise ValueError(f"context lacks required causal fields: {path}")
    state = torch.as_tensor(np.asarray(data["current_state"]), dtype=torch.float32, device=device)
    rain = torch.as_tensor(np.asarray(data["rainfall_scenarios"]), dtype=torch.float32, device=device)
    active = torch.as_tensor(np.asarray(data["active_target"]), dtype=torch.float32, device=device)
    flow = torch.as_tensor(np.asarray(data["previous_actuator_flow"]), dtype=torch.float32, device=device)
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state2 = state[0]
    elif state.ndim == 2:
        state2 = state
    else:
        raise ValueError(f"invalid current_state shape in {path}: {tuple(state.shape)}")
    if rain.ndim == 5 and int(rain.shape[0]) == 1:
        rain = rain[0]
    if rain.ndim != 4:
        raise ValueError(f"invalid rainfall shape in {path}: {tuple(rain.shape)}")
    active = active.reshape(-1)
    flow = flow.reshape(-1)
    if tuple(active.shape) != (109,) or tuple(flow.shape) != (109,):
        raise ValueError(f"invalid actuator context shape in {path}")
    return {"state": state2, "rain": rain, "active": active, "flow": flow}


def _decision_unit(row: dict[str, Any]) -> str:
    """Identify one causal state at which multiple candidate actions are compared."""
    raw = "|".join(
        (
            str(row["rainfall_group"]),
            str(row["query_set_id"]),
            str(row["context_npz_sha256"]),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V26 training requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    manifest_path = Path(args.dataset_manifest).resolve()
    records_path = Path(args.dataset_records).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _records(records_path)
    if str(manifest.get("records_sha256", "")).lower() != _sha(records_path).lower():
        raise ValueError("V26 dataset records SHA mismatch")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise ValueError("V26 dataset manifest record count mismatch")
    if set(str(row.get("split", "")) for row in rows) != {"train", "validation", "test"}:
        raise ValueError("V26 dataset must contain Train/Validation/Test")

    groups_by_split = {
        split: {str(row["rainfall_group"]) for row in rows if str(row["split"]) == split}
        for split in ("train", "validation", "test")
    }
    if groups_by_split["train"] & groups_by_split["validation"] or groups_by_split["train"] & groups_by_split["test"] or groups_by_split["validation"] & groups_by_split["test"]:
        raise ValueError("V26 rainfall groups leak across Train/Validation/Test")

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
    sources: list[str] = []
    stress: list[float] = []
    context_sha_failures = 0
    for row in rows:
        context_path = Path(str(row["context_npz"])).resolve()
        if not context_path.is_file():
            raise FileNotFoundError(context_path)
        expected_context_sha = str(row.get("context_npz_sha256", "")).lower()
        if len(expected_context_sha) == 64 and _sha(context_path).lower() != expected_context_sha:
            context_sha_failures += 1
            raise ValueError(f"V26 context SHA mismatch: {row['query_set_id']}")
        ctx = _context(context_path, device=device)
        candidate = torch.as_tensor(row["candidate_target"], dtype=torch.float32, device=device).reshape(-1)
        if tuple(candidate.shape) != (109,):
            raise ValueError(f"candidate target width mismatch: {row['query_set_id']}")
        with torch.inference_mode():
            built = build_v26_hydraulic_feature(
                step2_model=mpc.model,
                normalization=mpc.normalization,
                graph=mpc.graph,
                current_state=ctx["state"],
                rainfall_scenarios=ctx["rain"],
                previous_actuator_flow=ctx["flow"],
                active_target=ctx["active"],
                candidate_target=candidate,
                candidate_source=str(row["candidate_source"]),
                supervisory_mask=mpc.supervisory_mask,
                target_scale_m3=float(mpc.model.target_scale_m3.detach().cpu()),
            )
        features.append(built.feature.detach().cpu().numpy().astype(np.float64))
        truth.append(float(row["true_policy_return_delta_tfv_m3"]))
        splits.append(str(row["split"]))
        decision_units.append(_decision_unit(row))
        sources.append(str(row["candidate_source"]))
        stress.append(float(built.network_stress_q75))

    x = np.stack(features)
    y = np.asarray(truth, dtype=np.float64)
    split_array = np.asarray(splits)
    train = split_array == "train"
    validation = split_array == "validation"
    test = split_array == "test"
    if not train.any() or not validation.any() or not test.any():
        raise RuntimeError("V26 split lost one of Train/Validation/Test")

    model, training_report = fit_v26_value_model(
        x[train],
        y[train],
        x[validation],
        y[validation],
        [decision_units[index] for index in np.flatnonzero(validation)],
    )
    predictions = model.predict_numpy(x)
    split_reports: dict[str, Any] = {}
    for split, mask in (("train", train), ("validation", validation), ("test", test)):
        local_units = [decision_units[index] for index in np.flatnonzero(mask)]
        split_reports[split] = {
            "candidate_metrics": candidate_metrics(predictions[mask], y[mask]),
            "decision_metrics": decision_metrics(predictions[mask], y[mask], local_units),
            "record_count": int(mask.sum()),
            "decision_unit_count": len(set(local_units)),
            "rainfall_group_count": len(groups_by_split[split]),
        }

    control_payload = json.loads(Path(control_path).read_text(encoding="utf-8"))
    lineage = {
        "base_step2_sha256": _sha(step2_path),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "supervisory_mask_sha256": str(control_payload["supervisory_mask_sha256"]),
        "sequence_support_sha256": _sha(support_path),
        "dataset_manifest_sha256": _sha(manifest_path),
        "dataset_records_sha256": _sha(records_path),
        "feature_contract": V26_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "split_contract": str(manifest["contract"]),
        "decision_unit_contract": "SHA256_RAINFALL_GROUP_QUERY_CONTEXT_SHA",
        "v23_portfolio_contract": str(parent_lineage["v23_portfolio_contract"]),
        "v23_hydraulic_candidate_contract": str(parent_lineage["v23_hydraulic_candidate_contract"]),
        "v15_rank_checkpoint_sha256_parent_compatibility_only": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256_parent_compatibility_only": _sha(args.v21_boundary_checkpoint),
        "v15_rank_used_for_v26_candidate_selection": False,
        "v21_boundary_used_for_v26_action_admission": False,
    }

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"V26 training output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_report = split_reports["test"]
    payload = checkpoint_payload(
        model,
        lineage=lineage,
        training_report=training_report,
        test_report=test_report,
    )
    checkpoint_path = out_dir / "V26_HYDRAULIC_EXACT_RETURN_VALUE_MODEL.pt"
    torch.save(payload, checkpoint_path)

    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    stress_array = np.asarray(stress, dtype=np.float64)
    report = {
        "contract": V26_TRAINING_CONTRACT,
        "model_contract": V26_VALUE_MODEL_CONTRACT,
        "feature_contract": V26_HYDRAULIC_FEATURE_CONTRACT,
        "truth_field": "true_policy_return_delta_tfv_m3",
        "h120_role": "diagnostic_only",
        "dataset_manifest": str(manifest_path),
        "dataset_records": str(records_path),
        "feature_width": int(x.shape[1]),
        "record_count": int(len(rows)),
        "decision_unit_count": len(set(decision_units)),
        "candidate_source_counts": source_counts,
        "split_reports": split_reports,
        "validation_model_selection": training_report,
        "context_support": {
            "stress_min": float(stress_array.min()),
            "stress_median": float(np.median(stress_array)),
            "stress_max": float(stress_array.max()),
            "stress_std": float(stress_array.std()),
        },
        "lineage": lineage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "context_sha_failures": int(context_sha_failures),
        "scientific_metrics_block_runtime": False,
        "runtime_selection_rule": "MIN_PREDICTED_EXACT_RETURN_ACROSS_CANDIDATES_AND_HOLD_ZERO",
        "development_only": True,
        "formal_evidence": False,
        "ready_for_policy_lock": False,
    }
    report_path = out_dir / "V26_HYDRAULIC_EXACT_RETURN_VALUE_MODEL_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
