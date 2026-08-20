"""Run a no-SWMM early learnability gate before completing the expensive policy-return bulk.

This is a Development-only compute stop/go diagnostic. It reuses the exact current dataset compiler
and decision-aligned critic trainer, but temporarily lowers only the *pilot* training-group floor to
12 groups inside this process. The authoritative production/runtime contract remains unchanged:
48 train / 12 model-selection validation / 24 calibration groups are still required by the normal
learning pipeline and runtime checkpoint loader.

Use this gate only after at least 12 role-pure policy_return_train groups and all 12 frozen
policy_return_validation groups already exist. It never calls SWMM, never reads calibration,
Development-probe, Validation/Final/Formal truth outside those assigned learning roles, and never
produces a runtime-eligible checkpoint.

Interpretation:
- if decision-aligned fine-tuning beats epoch 0 on the frozen 12-group model-selection validation,
  the expensive bulk has demonstrated learnability and may continue;
- if it does not, stop new authoritative label generation and diagnose the critic/estimand/data
  alignment before spending the remaining SWMM budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterator

from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_learning_record


EARLY_GATE_CONTRACT = "PROJECT7_POLICY_RETURN_EARLY_LEARNABILITY_GATE_V1_NO_SWMM"
PILOT_MIN_TRAIN_GROUPS = 12
FROZEN_VALIDATION_GROUPS = 12
FULL_TRAIN_GROUPS = 48


@contextmanager
def _argv(values: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _role_groups(path: Path, role: str) -> set[str]:
    groups: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        validate_policy_return_learning_record(row)
        if str(row.get("data_role", "")) != role:
            raise ValueError(f"{path}:{line_number}: expected role {role}")
        groups.add(str(row["rainfall_group"]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--validation-records", required=True)
    parser.add_argument("--base-step2", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainable-scope", choices=("control-heads", "all"), default="control-heads")
    args = parser.parse_args()

    train_records = Path(args.train_records).resolve()
    validation_records = Path(args.validation_records).resolve()
    base_step2 = Path(args.base_step2).resolve()
    graph = Path(args.graph).resolve()
    for path in (train_records, validation_records, base_step2, graph):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_groups = _role_groups(train_records, "policy_return_train")
    validation_groups = _role_groups(validation_records, "policy_return_validation")
    if len(train_groups) < PILOT_MIN_TRAIN_GROUPS:
        raise ValueError(
            f"early learnability gate requires >= {PILOT_MIN_TRAIN_GROUPS} train rainfall groups; "
            f"got {len(train_groups)}"
        )
    if len(train_groups) >= FULL_TRAIN_GROUPS:
        raise ValueError("48 train groups already exist; run the normal learning pipeline instead")
    if len(validation_groups) < FROZEN_VALIDATION_GROUPS:
        raise ValueError(
            f"early learnability gate requires all {FROZEN_VALIDATION_GROUPS} validation rainfall "
            f"groups; got {len(validation_groups)}"
        )
    if train_groups & validation_groups:
        raise ValueError("early learnability gate detected train/validation rainfall-group overlap")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    compiler = _load_script(
        scripts / "compile_direct_tfv_policy_return_dataset_current.py",
        "policy_return_pilot_compiler",
    )
    trainer = _load_script(
        scripts / "train_direct_tfv_policy_return_current.py",
        "policy_return_pilot_trainer",
    )

    # Pilot-only relaxation. The normal compiler/trainer/runtime constants are not modified on disk.
    compiler._MIN_GROUPS["policy_return_train"] = PILOT_MIN_TRAIN_GROUPS
    trainer.DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS = PILOT_MIN_TRAIN_GROUPS

    train_dataset = out / "policy_return_train_pilot.npz"
    validation_dataset = out / "policy_return_validation_pilot.npz"
    pilot_checkpoint = out / "policy_return_critic_pilot_nonruntime.pt"

    with _argv(
        [
            str(scripts / "compile_direct_tfv_policy_return_dataset_current.py"),
            "--records-jsonl",
            str(train_records),
            "--data-role",
            "policy_return_train",
            "--out",
            str(train_dataset),
        ]
    ):
        compiler.main()
    with _argv(
        [
            str(scripts / "compile_direct_tfv_policy_return_dataset_current.py"),
            "--records-jsonl",
            str(validation_records),
            "--data-role",
            "policy_return_validation",
            "--out",
            str(validation_dataset),
        ]
    ):
        compiler.main()
    with _argv(
        [
            str(scripts / "train_direct_tfv_policy_return_current.py"),
            "--base-step2",
            str(base_step2),
            "--graph",
            str(graph),
            "--train-dataset",
            str(train_dataset),
            "--validation-dataset",
            str(validation_dataset),
            "--out",
            str(pilot_checkpoint),
            "--device",
            str(args.device),
            "--epochs",
            str(int(args.epochs)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--seed",
            str(int(args.seed)),
            "--trainable-scope",
            str(args.trainable_scope),
        ]
    ):
        trainer.main()

    report_path = pilot_checkpoint.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    improved = bool(report.get("fine_tuning_improved_over_epoch0", False))
    verdict = (
        "EARLY_POLICY_RETURN_LEARNABILITY_SUPPORTED_CONTINUE_FROZEN_BULK"
        if improved
        else "EARLY_POLICY_RETURN_LEARNABILITY_NOT_SUPPORTED_STOP_NEW_SWMM_LABELS"
    )
    summary = {
        "contract": EARLY_GATE_CONTRACT,
        "development_diagnostic_only": True,
        "runtime_checkpoint_eligible": False,
        "swmm_called_by_gate": False,
        "train_rainfall_group_count": len(train_groups),
        "validation_rainfall_group_count": len(validation_groups),
        "full_train_group_requirement_unchanged": FULL_TRAIN_GROUPS,
        "full_calibration_group_requirement_unchanged": 24,
        "validation_selected_epoch": report.get("validation_selected_epoch"),
        "fine_tuning_improved_over_epoch0": improved,
        "validation_baseline_metrics": report.get("validation_baseline_metrics", {}),
        "validation_metrics": report.get("validation_metrics", {}),
        "selection_rule": report.get("selection_rule"),
        "verdict": verdict,
        "next_action": (
            "CONTINUE_ONLY_MISSING_FROZEN_TRAIN_GROUPS_THEN_CALIBRATION"
            if improved
            else "STOP_BULK; DIAGNOSE_EXACT_POLICY_RETURN_CRITIC_GENERALIZATION; DO_NOT_RETRAIN_STEP2_BY_DEFAULT"
        ),
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }
    summary_path = out / "POLICY_RETURN_EARLY_LEARNABILITY_GATE.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
