"""Select the policy-return adaptation scope on frozen model-selection Validation before calibration.

This is the one-shot no-SWMM diagnosis for the current Project7 critic bottleneck.  The full exact
48/12/24 truth already exists.  The observed control-head trajectory can improve candidate ranking
while flipping globally from execute-all to HOLD-all, so before introducing a new model architecture
we test the smallest falsifiable representation hypothesis: are the frozen global state/rainfall
representations preventing exact-return decision adaptation?

Exactly two precommitted scopes are compared with identical train/validation data and hyperparameters:
``control-heads`` (the previous default) and ``all`` (the existing full-model fine-tuning ablation).
Calibration records are deliberately not compiled, scored or calibrated until one scope passes the
same decision-deployability gate used by the paper-facing learning pipeline.  If neither scope is
deployable, this command fails closed and declares the next diagnosis structural; do not keep tuning
against the same 12-group model-selection Validation set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file


SCOPE_SELECTION_CONTRACT = "PROJECT7_POLICY_RETURN_ONE_SHOT_ADAPTATION_SCOPE_SELECTION_V1_NO_SWMM"
PRECOMMITTED_SCOPES = ("control-heads", "all")


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print(json.dumps({"stage_command": command}, indent=2), flush=True)
    subprocess.run(command, check=True, env=env)


def _deployability(report: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(report.get("validation_metrics", {}))
    predicted_hold = float(metrics.get("predicted_hold_fraction", 1.0))
    oracle_hold = float(metrics.get("oracle_hold_optimal_fraction", 1.0))
    decision_accuracy = float(metrics.get("hold_aware_decision_accuracy", 0.0))
    action_starvation = bool(
        report.get("validation_action_starvation_detected", False)
        or (predicted_hold >= 1.0 - 1.0e-12 and oracle_hold < 1.0 - 1.0e-12)
    )
    decision_improved = bool(
        report.get("fine_tuning_improved_decision_metrics_over_epoch0", False)
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


def _selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    raw = report.get("validation_selection_key")
    if not isinstance(raw, list) or not raw:
        raise ValueError("policy-return checkpoint report lacks validation_selection_key")
    key = tuple(float(value) for value in raw)
    if not all(value == value and abs(value) != float("inf") for value in key):
        raise ValueError("policy-return validation_selection_key contains non-finite values")
    return key


def _select_scope(reports: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    if tuple(reports) != PRECOMMITTED_SCOPES:
        raise ValueError("scope-selection reports do not match the precommitted scope order")
    diagnostics: dict[str, Any] = {}
    eligible: list[tuple[tuple[float, ...], int, str]] = []
    for order, scope in enumerate(PRECOMMITTED_SCOPES):
        report = reports[scope]
        deployability = _deployability(report)
        diagnostics[scope] = {
            "validation_selected_epoch": report.get("validation_selected_epoch"),
            "validation_selection_key": list(_selection_key(report)),
            "validation_baseline_metrics": report.get("validation_baseline_metrics", {}),
            "validation_metrics": report.get("validation_metrics", {}),
            "fine_tuning_improved_over_epoch0": report.get("fine_tuning_improved_over_epoch0"),
            "fine_tuning_improved_decision_metrics_over_epoch0": report.get(
                "fine_tuning_improved_decision_metrics_over_epoch0"
            ),
            "validation_action_starvation_detected": report.get(
                "validation_action_starvation_detected"
            ),
            "deployability": deployability,
        }
        if deployability["passed"]:
            eligible.append((_selection_key(report), order, scope))
    if not eligible:
        return None, diagnostics
    eligible.sort()
    return eligible[0][2], diagnostics


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
    parser.add_argument("--coverage", type=float, default=0.90)
    args = parser.parse_args()
    if args.epochs <= 0 or not 0.0 < float(args.learning_rate) < 1.0e-2:
        raise ValueError("invalid policy-return scope-selection training hyperparameters")
    if abs(float(args.coverage) - 0.90) > 1.0e-12:
        raise ValueError("paper-facing policy-return admission coverage remains frozen at 0.90")

    inputs = {
        "train_records": Path(args.train_records).resolve(),
        "validation_records": Path(args.validation_records).resolve(),
        "calibration_records": Path(args.calibration_records).resolve(),
        "base_step2": Path(args.base_step2).resolve(),
        "graph": Path(args.graph).resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    selection_root = out / "scope_selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    train_dataset = selection_root / "policy_return_train.npz"
    validation_dataset = selection_root / "policy_return_validation.npz"

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    compile_script = scripts / "compile_direct_tfv_policy_return_dataset_current.py"
    for records, role, destination in (
        (inputs["train_records"], "policy_return_train", train_dataset),
        (inputs["validation_records"], "policy_return_validation", validation_dataset),
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
            env=env,
        )

    reports: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, Path] = {}
    for scope in PRECOMMITTED_SCOPES:
        scope_dir = selection_root / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = scope_dir / "policy_return_critic.pt"
        _run(
            [
                sys.executable,
                str(scripts / "train_direct_tfv_policy_return_current.py"),
                "--base-step2",
                str(inputs["base_step2"]),
                "--graph",
                str(inputs["graph"]),
                "--train-dataset",
                str(train_dataset),
                "--validation-dataset",
                str(validation_dataset),
                "--out",
                str(checkpoint),
                "--device",
                str(args.device),
                "--epochs",
                str(int(args.epochs)),
                "--learning-rate",
                str(float(args.learning_rate)),
                "--seed",
                str(int(args.seed)),
                "--trainable-scope",
                scope,
            ],
            env=env,
        )
        report_path = checkpoint.with_suffix(".json")
        if not report_path.is_file():
            raise RuntimeError(f"{scope}: trainer did not emit checkpoint report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if str(report.get("trainable_scope", "")) != scope:
            raise RuntimeError(f"{scope}: checkpoint report has another trainable scope")
        reports[scope] = report
        checkpoint_paths[scope] = checkpoint

    selected_scope, diagnostics = _select_scope(reports)
    selection_summary = {
        "contract": SCOPE_SELECTION_CONTRACT,
        "development_model_selection_only": True,
        "swmm_called_by_scope_selection": False,
        "precommitted_scopes": list(PRECOMMITTED_SCOPES),
        "same_hyperparameters_across_scopes": True,
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "selection_basis": "FROZEN_12_GROUP_MODEL_SELECTION_VALIDATION_ONLY",
        "calibration_used_for_scope_selection": False,
        "calibration_scored_before_scope_selection": False,
        "train_records_sha256": sha256_file(inputs["train_records"]),
        "validation_records_sha256": sha256_file(inputs["validation_records"]),
        "calibration_records_sha256": sha256_file(inputs["calibration_records"]),
        "scope_diagnostics": diagnostics,
        "selected_trainable_scope": selected_scope,
        "ready_for_selected_pipeline": selected_scope is not None,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
        "if_no_scope_passes": (
            "STOP_NO_SWMM; STRUCTURAL_QUERY_CONDITIONED_HOLD_MARGIN_OR_REPRESENTATION_"
            "BOTTLENECK; DO_NOT_KEEP_TUNING_AGAINST_THE_SAME_VALIDATION_SET"
        ),
    }
    selection_path = out / "POLICY_RETURN_ADAPTATION_SCOPE_SELECTION.json"
    selection_path.write_text(
        json.dumps(selection_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if selected_scope is None:
        raise RuntimeError(
            "neither precommitted adaptation scope passed decision deployability; calibration was not "
            "used; stop before pi1 and diagnose a structural query-conditioned HOLD-margin bottleneck"
        )

    selected_pipeline = out / "selected_pipeline"
    _run(
        [
            sys.executable,
            str(scripts / "run_direct_tfv_policy_return_learning_pipeline_current.py"),
            "--train-records",
            str(inputs["train_records"]),
            "--validation-records",
            str(inputs["validation_records"]),
            "--calibration-records",
            str(inputs["calibration_records"]),
            "--base-step2",
            str(inputs["base_step2"]),
            "--graph",
            str(inputs["graph"]),
            "--out-dir",
            str(selected_pipeline),
            "--device",
            str(args.device),
            "--epochs",
            str(int(args.epochs)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--seed",
            str(int(args.seed)),
            "--trainable-scope",
            selected_scope,
            "--coverage",
            str(float(args.coverage)),
        ],
        env=env,
    )
    pipeline_summary_path = selected_pipeline / "POLICY_RETURN_LEARNING_PIPELINE.json"
    if not pipeline_summary_path.is_file():
        raise RuntimeError("selected policy-return learning pipeline did not emit its summary")
    pipeline_summary = json.loads(pipeline_summary_path.read_text(encoding="utf-8"))
    if pipeline_summary.get("ready_for_pi1_development") is not True:
        raise RuntimeError("selected adaptation scope did not remain deployable in the full pipeline")

    final_summary = dict(selection_summary)
    final_summary.update(
        {
            "selected_pipeline_summary": str(pipeline_summary_path),
            "selected_pipeline_summary_sha256": sha256_file(pipeline_summary_path),
            "selected_checkpoint_path": str(
                (selected_pipeline / "policy_return_critic.pt").resolve()
            ),
            "selected_admission_path": str(
                (selected_pipeline / "policy_return_admission.json").resolve()
            ),
            "calibration_scored_after_scope_selection": True,
            "ready_for_pi1_development": True,
            "next_stage": "RUN_FROZEN_THREE_EVENT_PI1_DEVELOPMENT; READY_FOR_POLICY_LOCK_REMAINS_FALSE",
        }
    )
    selection_path.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
