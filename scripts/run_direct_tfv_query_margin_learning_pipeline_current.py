"""Run the no-SWMM query-conditioned policy-return learning/calibration chain.

Both the old 12-group Validation and old 24-group Calibration have been exposed by the retired scalar
critic workflow. They are therefore development diagnostics only for this redesigned critic. The new
path requires fresh rainfall-group-disjoint Validation and fresh Calibration. The fixed rank+margin
critic is trained only on the original 48 Train groups, evaluated once on fresh Validation, and only
then scored/calibrated on fresh Calibration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from rtc.direct_tfv_policy_return import sha256_file
from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_learning_record


def _rows(path: Path, role: str) -> list[dict]:
    rows = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: row is not an object")
        validate_policy_return_learning_record(row)
        if str(row.get("data_role")) != role:
            raise ValueError(f"{path}:{n}: expected {role}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{role} records are empty")
    return rows


def _run(command: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"stage_command": command}, indent=2), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-records", required=True)
    p.add_argument("--fresh-validation-records", required=True)
    p.add_argument("--deprecated-validation-records", required=True)
    p.add_argument("--fresh-calibration-records", required=True)
    p.add_argument("--deprecated-calibration-records", required=True)
    p.add_argument("--base-step2", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--supervisory-control", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--coverage", type=float, default=0.90)
    args = p.parse_args()
    if abs(float(args.coverage) - 0.90) > 1.0e-12:
        raise ValueError("paper-facing conformal coverage remains frozen at 0.90")
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "train": args.train_records,
            "fresh_validation": args.fresh_validation_records,
            "deprecated_validation": args.deprecated_validation_records,
            "fresh_calibration": args.fresh_calibration_records,
            "deprecated_calibration": args.deprecated_calibration_records,
            "base_step2": args.base_step2,
            "graph": args.graph,
            "supervisory_control": args.supervisory_control,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train = _rows(paths["train"], "policy_return_train")
    fresh_validation = _rows(paths["fresh_validation"], "policy_return_validation")
    old_validation = _rows(paths["deprecated_validation"], "policy_return_validation")
    fresh_calibration = _rows(paths["fresh_calibration"], "policy_return_calibration")
    old_calibration = _rows(paths["deprecated_calibration"], "policy_return_calibration")
    groups = {
        "train": {str(r["rainfall_group"]) for r in train},
        "fresh_validation": {str(r["rainfall_group"]) for r in fresh_validation},
        "deprecated_validation": {str(r["rainfall_group"]) for r in old_validation},
        "fresh_calibration": {str(r["rainfall_group"]) for r in fresh_calibration},
        "deprecated_calibration": {str(r["rainfall_group"]) for r in old_calibration},
    }
    if len(groups["train"]) < 48 or len(groups["fresh_validation"]) < 12 or len(groups["fresh_calibration"]) < 24:
        raise ValueError("query-margin pipeline requires 48 train / 12 fresh validation / 24 fresh calibration groups")
    forbidden_pairs = (
        ("train", "fresh_validation"),
        ("train", "fresh_calibration"),
        ("fresh_validation", "fresh_calibration"),
        ("fresh_validation", "deprecated_validation"),
        ("fresh_validation", "deprecated_calibration"),
        ("fresh_calibration", "deprecated_validation"),
        ("fresh_calibration", "deprecated_calibration"),
    )
    for left, right in forbidden_pairs:
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ValueError(f"query-margin role overlap {left}/{right}: {overlap}")
    lineage_keys = ("continuation_policy_sha256", "supervisory_mask_sha256", "candidate_portfolio_contract")
    learning_rows = train + fresh_validation + fresh_calibration
    lineage = {key: {str(r[key]).lower() for r in learning_rows} for key in lineage_keys}
    if any(len(values) != 1 for values in lineage.values()):
        raise ValueError("query-margin fresh roles mix policy/control/portfolio lineage")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    compile_script = scripts / "compile_direct_tfv_policy_return_dataset_current.py"
    datasets = {
        "train": out / "policy_return_train.npz",
        "fresh_validation": out / "policy_return_fresh_validation.npz",
        "deprecated_validation": out / "policy_return_deprecated_validation.npz",
    }
    for records, role, destination in (
        (paths["train"], "policy_return_train", datasets["train"]),
        (paths["fresh_validation"], "policy_return_validation", datasets["fresh_validation"]),
        (paths["deprecated_validation"], "policy_return_validation", datasets["deprecated_validation"]),
    ):
        _run(
            [
                sys.executable,
                str(compile_script),
                "--records-jsonl",
                str(records),
                "--data-role",
                role,
                "--out",
                str(destination),
            ],
            env,
        )
    checkpoint = out / "policy_return_query_margin_critic.pt"
    _run(
        [
            sys.executable,
            str(scripts / "train_direct_tfv_policy_return_query_margin_current.py"),
            "--base-step2",
            str(paths["base_step2"]),
            "--graph",
            str(paths["graph"]),
            "--supervisory-control",
            str(paths["supervisory_control"]),
            "--train-dataset",
            str(datasets["train"]),
            "--fresh-validation-dataset",
            str(datasets["fresh_validation"]),
            "--deprecated-validation-dataset",
            str(datasets["deprecated_validation"]),
            "--out",
            str(checkpoint),
            "--device",
            args.device,
            "--seed",
            "42",
        ],
        env,
    )
    report = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if report.get("fresh_validation_verified") is not True or report.get("ready_for_calibration") is not True:
        raise RuntimeError("fresh-validation gate did not authorize calibration")
    scored = out / "policy_return_fresh_calibration_query_margin_scored.jsonl"
    _run(
        [
            sys.executable,
            str(scripts / "score_direct_tfv_policy_return_query_margin_calibration_current.py"),
            "--records-jsonl",
            str(paths["fresh_calibration"]),
            "--query-margin-checkpoint",
            str(checkpoint),
            "--base-step2",
            str(paths["base_step2"]),
            "--graph",
            str(paths["graph"]),
            "--supervisory-control",
            str(paths["supervisory_control"]),
            "--out",
            str(scored),
            "--device",
            args.device,
        ],
        env,
    )
    admission = out / "policy_return_query_margin_admission.json"
    _run(
        [
            sys.executable,
            str(scripts / "calibrate_direct_tfv_policy_return_portfolio_admission_current.py"),
            "--records-jsonl",
            str(scored),
            "--policy-return-checkpoint",
            str(checkpoint),
            "--continuation-policy-sha256",
            next(iter(lineage["continuation_policy_sha256"])),
            "--out",
            str(admission),
            "--coverage",
            "0.90",
        ],
        env,
    )
    admission_payload = json.loads(admission.read_text(encoding="utf-8"))
    if float(admission_payload.get("coverage", float("nan"))) != 0.90:
        raise RuntimeError("admission coverage drifted")
    summary = {
        "contract": "PROJECT7_QUERY_CONDITIONED_POLICY_RETURN_LEARNING_PIPELINE_V2_FRESH_VAL_CAL",
        "complete": True,
        "swmm_called_by_pipeline": False,
        "step1_retrained": False,
        "base_step2_retrained": False,
        "train_group_count": len(groups["train"]),
        "fresh_validation_group_count": len(groups["fresh_validation"]),
        "fresh_calibration_group_count": len(groups["fresh_calibration"]),
        "deprecated_validation_group_count": len(groups["deprecated_validation"]),
        "deprecated_calibration_group_count": len(groups["deprecated_calibration"]),
        "fresh_validation_disjoint_from_consumed_evidence": True,
        "fresh_calibration_disjoint_from_consumed_evidence": True,
        "fresh_calibration_used_for_training_or_model_selection": False,
        "fresh_validation_metrics": report["fresh_validation_metrics"],
        "fresh_validation_thresholds": report["fresh_validation_thresholds"],
        "fresh_validation_verified": True,
        "policy_return_checkpoint_sha256": sha256_file(checkpoint),
        "policy_return_admission_sha256": sha256_file(admission),
        "coverage": 0.90,
        "normalized_residual_conformal_upper": admission_payload.get("normalized_residual_conformal_upper"),
        "ready_for_pi1_development": True,
        "ready_for_policy_lock": False,
        "next_stage": "RUN_ONE_FROZEN_PI1_DEVELOPMENT_EVENT; CONTINUE_TO_THREE_EVENTS_ONLY_IF_NONDEGENERATE_AND_ENGINEERING_PASS",
    }
    summary_path = out / "QUERY_MARGIN_LEARNING_PIPELINE.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
