"""Run the complete no-SWMM policy-return learning chain after exact truth is frozen.

This orchestrator deliberately starts at authoritative role-pure JSONL records. Expensive SWMM parent
and exact-query generation remain outside this process and must already have passed the current truth
firewall. The pipeline then performs, in one fail-closed command:

1. role and lineage audit for train / model-selection validation / calibration;
2. role-pure dataset compilation;
3. decision-aligned critic fine-tuning with epoch-0 baseline preservation;
4. frozen-critic scoring of untouched calibration records;
5. matched one-sided split-conformal admission;
6. a validation deployability gate that rejects trivial all-HOLD/action-starvation solutions;
7. an immutable summary for the subsequent pi1 Development run.

The script never calls SWMM, never reads Development/Final truth, never weakens q95 support or
conformal coverage, and never promotes Policy Lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS,
    DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS,
    sha256_file,
)
from rtc.direct_tfv_policy_return_portfolio_admission import (
    CURRENT_THREE_FAMILY_SOURCES,
    validate_policy_return_learning_record,
)


PIPELINE_CONTRACT = "PROJECT7_DIRECT_TFV_POLICY_RETURN_LEARNING_PIPELINE_V2_DECISION_DEPLOYABLE"
ROLE_MINIMUMS = {
    "policy_return_train": DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    "policy_return_validation": DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS,
    "policy_return_calibration": DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS,
}


def _read_role(path: str | Path, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        validate_policy_return_learning_record(row)
        if str(row.get("data_role", "")) != role:
            raise ValueError(f"{path}:{line_number}: expected role {role}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{role} records are empty")
    groups = {str(row["rainfall_group"]) for row in rows}
    if len(groups) < ROLE_MINIMUMS[role]:
        raise ValueError(
            f"{role} requires >= {ROLE_MINIMUMS[role]} independent rainfall groups; got {len(groups)}"
        )
    return rows


def _audit_roles(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
) -> dict[str, Any]:
    by_role = {
        "policy_return_train": train,
        "policy_return_validation": validation,
        "policy_return_calibration": calibration,
    }
    groups = {
        role: {str(row["rainfall_group"]) for row in rows}
        for role, rows in by_role.items()
    }
    for left, right in (
        ("policy_return_train", "policy_return_validation"),
        ("policy_return_train", "policy_return_calibration"),
        ("policy_return_validation", "policy_return_calibration"),
    ):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ValueError(f"policy-return role overlap {left}/{right}: {overlap}")

    all_rows = train + validation + calibration
    continuation = {str(row["continuation_policy_sha256"]).lower() for row in all_rows}
    masks = {str(row["supervisory_mask_sha256"]).lower() for row in all_rows}
    if len(continuation) != 1:
        raise ValueError("learning pipeline mixes continuation-policy lineages")
    if len(masks) != 1:
        raise ValueError("learning pipeline mixes supervisory-control masks")
    if {int(row.get("supervisory_control_dimension", -1)) for row in all_rows} != {82}:
        raise ValueError("learning pipeline requires the frozen 82-control subspace")
    if {int(row.get("model_action_channel_count", -1)) for row in all_rows} != {109}:
        raise ValueError("learning pipeline requires the frozen 109-channel representation")
    sources = {str(row["candidate_source"]) for row in all_rows}
    if not sources.issubset(set(CURRENT_THREE_FAMILY_SOURCES)):
        raise ValueError("learning pipeline contains a non-current candidate family")
    if "TYPE_AWARE_HYDRAULIC_PRESSURE" not in sources:
        raise ValueError("learning pipeline lacks the type-aware hydraulic family")
    if not any(source.startswith("STEP2_H10_PROBE_SCALE_") for source in sources):
        raise ValueError("learning pipeline lacks a supported Step2 H10-probe family")

    return {
        "continuation_policy_sha256": next(iter(continuation)),
        "supervisory_mask_sha256": next(iter(masks)),
        "role_group_counts": {role: len(values) for role, values in groups.items()},
        "role_record_counts": {role: len(rows) for role, rows in by_role.items()},
        "candidate_source_counts": dict(
            sorted(Counter(str(row["candidate_source"]) for row in all_rows).items())
        ),
        "role_disjoint": True,
        "authoritative_truth_firewall_verified": True,
        "development_diagnostic_rows_allowed": False,
    }


def _validation_deployability(checkpoint_report: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(checkpoint_report.get("validation_metrics", {}))
    predicted_hold = float(metrics.get("predicted_hold_fraction", 1.0))
    oracle_hold = float(metrics.get("oracle_hold_optimal_fraction", 1.0))
    decision_accuracy = float(metrics.get("hold_aware_decision_accuracy", 0.0))
    action_starvation = bool(
        checkpoint_report.get("validation_action_starvation_detected", False)
        or (predicted_hold >= 1.0 - 1.0e-12 and oracle_hold < 1.0 - 1.0e-12)
    )
    decision_improved = bool(
        checkpoint_report.get("fine_tuning_improved_decision_metrics_over_epoch0", False)
    )
    passed = bool(decision_improved and not action_starvation and decision_accuracy > 0.0)
    return {
        "passed": passed,
        "fine_tuning_improved_decision_metrics_over_epoch0": decision_improved,
        "validation_action_starvation_detected": action_starvation,
        "validation_predicted_hold_fraction": predicted_hold,
        "validation_oracle_hold_optimal_fraction": oracle_hold,
        "validation_hold_aware_decision_accuracy": decision_accuracy,
    }


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print(json.dumps({"stage_command": command}, indent=2))
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--validation-records", required=True)
    parser.add_argument("--calibration-records", required=True)
    parser.add_argument("--base-step2", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainable-scope", choices=("control-heads", "all"), default="control-heads")
    parser.add_argument("--coverage", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.5 < float(args.coverage) < 1.0:
        raise ValueError("conformal coverage must lie in (0.5,1)")
    if abs(float(args.coverage) - 0.90) > 1.0e-12:
        raise ValueError("current paper-facing policy-return admission coverage is frozen at 0.90")

    for input_path in (
        args.train_records,
        args.validation_records,
        args.calibration_records,
        args.base_step2,
        args.graph,
    ):
        if not Path(input_path).is_file():
            raise FileNotFoundError(input_path)

    train_rows = _read_role(args.train_records, "policy_return_train")
    validation_rows = _read_role(args.validation_records, "policy_return_validation")
    calibration_rows = _read_role(args.calibration_records, "policy_return_calibration")
    audit = _audit_roles(train_rows, validation_rows, calibration_rows)

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    train_dataset = out / "policy_return_train.npz"
    validation_dataset = out / "policy_return_validation.npz"
    calibration_dataset = out / "policy_return_calibration.npz"
    checkpoint = out / "policy_return_critic.pt"
    scored_calibration = out / "policy_return_calibration_scored.jsonl"
    admission = out / "policy_return_admission.json"

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    compile_script = scripts / "compile_direct_tfv_policy_return_dataset_current.py"
    for records, role, destination in (
        (args.train_records, "policy_return_train", train_dataset),
        (args.validation_records, "policy_return_validation", validation_dataset),
        (args.calibration_records, "policy_return_calibration", calibration_dataset),
    ):
        _run(
            [
                sys.executable,
                str(compile_script),
                "--records-jsonl",
                str(Path(records).resolve()),
                "--data-role",
                role,
                "--out",
                str(destination),
            ],
            env=env,
        )

    _run(
        [
            sys.executable,
            str(scripts / "train_direct_tfv_policy_return_current.py"),
            "--base-step2",
            str(Path(args.base_step2).resolve()),
            "--graph",
            str(Path(args.graph).resolve()),
            "--train-dataset",
            str(train_dataset),
            "--validation-dataset",
            str(validation_dataset),
            "--out",
            str(checkpoint),
            "--device",
            args.device,
            "--epochs",
            str(int(args.epochs)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--seed",
            str(int(args.seed)),
            "--trainable-scope",
            args.trainable_scope,
        ],
        env=env,
    )

    _run(
        [
            sys.executable,
            str(scripts / "score_direct_tfv_policy_return_calibration_current.py"),
            "--records-jsonl",
            str(Path(args.calibration_records).resolve()),
            "--policy-return-checkpoint",
            str(checkpoint),
            "--base-step2",
            str(Path(args.base_step2).resolve()),
            "--graph",
            str(Path(args.graph).resolve()),
            "--out",
            str(scored_calibration),
            "--device",
            args.device,
        ],
        env=env,
    )

    _run(
        [
            sys.executable,
            str(scripts / "calibrate_direct_tfv_policy_return_portfolio_admission_current.py"),
            "--records-jsonl",
            str(scored_calibration),
            "--policy-return-checkpoint",
            str(checkpoint),
            "--continuation-policy-sha256",
            str(audit["continuation_policy_sha256"]),
            "--out",
            str(admission),
            "--coverage",
            str(float(args.coverage)),
        ],
        env=env,
    )

    checkpoint_report_path = checkpoint.with_suffix(".json")
    if not checkpoint_report_path.is_file() or not admission.is_file():
        raise RuntimeError("learning pipeline did not produce checkpoint/admission artifacts")
    checkpoint_report = json.loads(checkpoint_report_path.read_text(encoding="utf-8"))
    admission_payload = json.loads(admission.read_text(encoding="utf-8"))
    if admission_payload.get("development_diagnostic_rows_allowed") is not False:
        raise RuntimeError("calibration artifact weakened the learning firewall")
    if float(admission_payload.get("coverage", float("nan"))) != 0.90:
        raise RuntimeError("calibration artifact changed the frozen 0.90 coverage")
    if str(admission_payload.get("supervisory_mask_sha256", "")).lower() != str(
        audit["supervisory_mask_sha256"]
    ).lower():
        raise RuntimeError("learning pipeline changed supervisory-control lineage")

    deployability = _validation_deployability(checkpoint_report)
    summary = {
        "contract": PIPELINE_CONTRACT,
        "complete": True,
        "swmm_called_by_pipeline": False,
        "authoritative_truth_preexisting": True,
        **audit,
        "base_step2_sha256": sha256_file(args.base_step2),
        "graph_sha256": sha256_file(args.graph),
        "train_records_sha256": sha256_file(args.train_records),
        "validation_records_sha256": sha256_file(args.validation_records),
        "calibration_records_sha256": sha256_file(args.calibration_records),
        "train_dataset_sha256": sha256_file(train_dataset),
        "validation_dataset_sha256": sha256_file(validation_dataset),
        "calibration_dataset_sha256": sha256_file(calibration_dataset),
        "policy_return_checkpoint_sha256": sha256_file(checkpoint),
        "scored_calibration_sha256": sha256_file(scored_calibration),
        "policy_return_admission_sha256": sha256_file(admission),
        "validation_selected_epoch": checkpoint_report.get("validation_selected_epoch"),
        "fine_tuning_improved_over_epoch0": checkpoint_report.get("fine_tuning_improved_over_epoch0"),
        "fine_tuning_improved_decision_metrics_over_epoch0": checkpoint_report.get(
            "fine_tuning_improved_decision_metrics_over_epoch0"
        ),
        "validation_baseline_metrics": checkpoint_report.get("validation_baseline_metrics", {}),
        "validation_metrics": checkpoint_report.get("validation_metrics", {}),
        "selection_rule": checkpoint_report.get("selection_rule"),
        "coverage": admission_payload.get("coverage"),
        "normalized_residual_conformal_upper": admission_payload.get(
            "normalized_residual_conformal_upper"
        ),
        "validation_deployability_gate": deployability,
        "ready_for_pi1_development": bool(deployability["passed"]),
        "ready_for_policy_lock": False,
        "next_stage": (
            "RUN_ROLE_DISJOINT_PI1_DEVELOPMENT_AND_OPERATIONAL_COMPARATORS; "
            "IF_PI1_SHIFTS_STATE_ACTION_DISTRIBUTION, COLLECT_ROLE_DISJOINT_Q_PI1_BEFORE_POLICY_LOCK"
            if deployability["passed"]
            else "STOP_BEFORE_PI1; CRITIC_DECISION_GENERALIZATION_OR_ACTION_STARVATION_REMAINS"
        ),
    }
    summary_path = out / "POLICY_RETURN_LEARNING_PIPELINE.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["pipeline_summary_sha256"] = sha256_file(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not deployability["passed"]:
        raise RuntimeError(
            "policy-return critic failed validation deployability gate; do not start pi1 Development"
        )


if __name__ == "__main__":
    main()
