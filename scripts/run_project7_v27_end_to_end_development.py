"""Run complete Project7 V27 Development workflow with one command.

Stages:
  0. inventory historical exact-return assets;
  1. build V27 leakage-safe dataset with composite context recovery;
  2. train decision-aware pairwise exact-return model with Train group-CV + Validation selection;
  3. run all five Proposed Benchmark5 events using immutable baselines.

No offline model-quality threshold can suppress Stage 3.  Only code/data-integrity or engineering
execution failures stop the workflow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _run(command: list[str]) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--context-records", action="append", default=[])
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.out_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"V27 end-to-end output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    inventory_path = root / "V27_EXACT_RETURN_HISTORY_INVENTORY.json"
    dataset_dir = root / "dataset"
    model_dir = root / "model"
    benchmark_dir = root / "benchmark5"

    _run(
        [
            sys.executable,
            str(scripts / "audit_project7_v26_exact_return_inventory.py"),
            "--root", str(Path(args.study_root).resolve()),
            "--out", str(inventory_path),
        ]
    )

    build = [
        sys.executable,
        str(scripts / "build_project7_v27_exact_return_dataset.py"),
        "--inventory", str(inventory_path),
        "--study-root", str(Path(args.study_root).resolve()),
        "--out-dir", str(dataset_dir),
        "--seed", str(int(args.seed)),
        "--train-fraction", str(float(args.train_fraction)),
        "--validation-fraction", str(float(args.validation_fraction)),
    ]
    for value in args.context_records:
        build.extend(("--context-records", str(Path(value).resolve())))
    _run(build)

    dataset_manifest = dataset_dir / "V27_EXACT_RETURN_DATASET_MANIFEST.json"
    dataset_records = dataset_dir / "V27_EXACT_RETURN_RECORDS.jsonl"
    _run(
        [
            sys.executable,
            str(scripts / "train_project7_v27_decision_value_current.py"),
            "--asset-manifest", str(Path(args.asset_manifest).resolve()),
            "--dataset-manifest", str(dataset_manifest),
            "--dataset-records", str(dataset_records),
            "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
            "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
            "--out-dir", str(model_dir),
            "--device", str(args.device),
            "--probe-chunk-size", str(int(args.probe_chunk_size)),
            "--seed", str(int(args.seed)),
            "--cv-folds", str(int(args.cv_folds)),
        ]
    )

    value_checkpoint = model_dir / "V27_DECISION_AWARE_EXACT_RETURN_VALUE_MODEL.pt"
    # Deliberately do not inspect AUC/precision/harmful-action metrics here. Benchmark5 is the
    # authoritative next Development experiment rather than a privilege granted by an offline gate.
    _run(
        [
            sys.executable,
            str(scripts / "run_project7_operational_benchmark5_v27_development.py"),
            "--benchmark-manifest", str(Path(args.benchmark_manifest).resolve()),
            "--baseline-cache", str(Path(args.baseline_cache).resolve()),
            "--asset-manifest", str(Path(args.asset_manifest).resolve()),
            "--priority-nodes", str(Path(args.priority_nodes).resolve()),
            "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
            "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
            "--v27-value-checkpoint", str(value_checkpoint),
            "--dataset-manifest", str(dataset_manifest),
            "--out-dir", str(benchmark_dir),
            "--device", str(args.device),
            "--decision-runtime-budget-seconds", str(float(args.decision_runtime_budget_seconds)),
            "--probe-chunk-size", str(int(args.probe_chunk_size)),
        ]
    )

    summary = {
        "contract": "PROJECT7_V27_END_TO_END_DEVELOPMENT_WORKFLOW_V1",
        "completed": True,
        "inventory": str(inventory_path),
        "dataset_manifest": str(dataset_manifest),
        "dataset_records": str(dataset_records),
        "value_model_report": str(model_dir / "V27_DECISION_AWARE_EXACT_RETURN_VALUE_MODEL_REPORT.json"),
        "value_checkpoint": str(value_checkpoint),
        "benchmark_root": str(benchmark_dir),
        "auto_rbc_shadow_candidate_enabled": True,
        "runtime_ranking_uses_unclipped_latent": True,
        "q95_execution_preserved": True,
        "q95_precontraction_counterfactual_scoring": True,
        "offline_scientific_gate_between_training_and_benchmark": False,
        "new_training_truth_swmm": 0,
        "baseline_rerun_requested": False,
        "development_only": True,
        "ready_for_policy_lock": False,
    }
    summary_path = root / "V27_END_TO_END_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
