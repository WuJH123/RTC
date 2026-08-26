"""Run the complete Project7 V26 Development flow with one command.

Stages are deliberately simple and sequential:
  0. inventory reusable historical candidate-vs-HOLD exact-return JSONL/JSON/NPZ assets;
  1. recover/canonicalize/adjudicate them and freeze a leakage-safe Train/Validation/Test split;
  2. train/select the V26 action-conditioned value model and report Test metrics;
  3. run all five Proposed Benchmark5 events while reusing immutable baselines.

No offline model-quality statistic can short-circuit Stage 3. Only an actual program/data-lineage or
engineering-execution failure stops the workflow. Historical role/version exposure is provenance,
not a training exclusion rule. Genuinely ambiguous individual truth keys may be quarantined by the
dataset builder without discarding unrelated valid SWMM supervision.
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
    parser.add_argument(
        "--study-root",
        help="Project7 study tree containing historical JSONL/JSON/NPZ exact-return assets",
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Additional explicit historical JSONL/JSON/NPZ asset; may be repeated",
    )
    parser.add_argument(
        "--records-jsonl",
        action="append",
        default=[],
        help="Backward-compatible alias for --asset",
    )
    parser.add_argument(
        "--context-records",
        action="append",
        default=[],
        help="Backward-compatible context-reference JSONL used to repair historical rows",
    )
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
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if not args.study_root and not args.asset and not args.records_jsonl:
        raise ValueError("V26 requires --study-root and/or at least one explicit historical asset")

    root = Path(args.out_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"V26 end-to-end output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    dataset_dir = root / "dataset"
    model_dir = root / "model"
    benchmark_dir = root / "benchmark5"

    inventory_path: Path | None = None
    if args.study_root:
        inventory_path = root / "V26_EXACT_RETURN_HISTORY_INVENTORY.json"
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
        str(scripts / "build_project7_v26_exact_return_dataset.py"),
    ]
    if inventory_path is not None:
        build.extend(("--inventory", str(inventory_path)))
        build.extend(("--study-root", str(Path(args.study_root).resolve())))
    for value in list(args.asset) + list(args.records_jsonl):
        build.extend(("--asset", str(Path(value).resolve())))
    for value in args.context_records:
        build.extend(("--context-records", str(Path(value).resolve())))
    build.extend(
        (
            "--out-dir", str(dataset_dir),
            "--seed", str(int(args.seed)),
            "--train-fraction", str(float(args.train_fraction)),
            "--validation-fraction", str(float(args.validation_fraction)),
        )
    )
    _run(build)

    dataset_manifest = dataset_dir / "V26_EXACT_RETURN_DATASET_MANIFEST.json"
    dataset_records = dataset_dir / "V26_EXACT_RETURN_RECORDS.jsonl"
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    leakage = dataset_payload.get("leakage_audit", {})
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise RuntimeError("V26 dataset leakage audit did not pass")

    _run(
        [
            sys.executable,
            str(scripts / "train_project7_v26_hydraulic_value_current.py"),
            "--asset-manifest", str(Path(args.asset_manifest).resolve()),
            "--dataset-manifest", str(dataset_manifest),
            "--dataset-records", str(dataset_records),
            "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
            "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
            "--out-dir", str(model_dir),
            "--device", str(args.device),
            "--probe-chunk-size", str(int(args.probe_chunk_size)),
        ]
    )

    value_checkpoint = model_dir / "V26_HYDRAULIC_EXACT_RETURN_VALUE_MODEL.pt"
    # Benchmark5 is the next experiment, not a privilege granted by an arbitrary offline score gate.
    _run(
        [
            sys.executable,
            str(scripts / "run_project7_operational_benchmark5_v26_development.py"),
            "--benchmark-manifest", str(Path(args.benchmark_manifest).resolve()),
            "--baseline-cache", str(Path(args.baseline_cache).resolve()),
            "--asset-manifest", str(Path(args.asset_manifest).resolve()),
            "--priority-nodes", str(Path(args.priority_nodes).resolve()),
            "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
            "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
            "--v26-value-checkpoint", str(value_checkpoint),
            "--dataset-manifest", str(dataset_manifest),
            "--out-dir", str(benchmark_dir),
            "--device", str(args.device),
            "--decision-runtime-budget-seconds", str(float(args.decision_runtime_budget_seconds)),
            "--probe-chunk-size", str(int(args.probe_chunk_size)),
        ]
    )

    inventory_payload = None
    if inventory_path is not None:
        inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = {
        "contract": "PROJECT7_V26_END_TO_END_DEVELOPMENT_WORKFLOW_V4",
        "completed": True,
        "historical_inventory": str(inventory_path) if inventory_path is not None else None,
        "inventory_candidate_exact_rows_before_dedup": (
            inventory_payload.get("candidate_exact_return_rows_before_canonicalization_and_dedup")
            if inventory_payload else None
        ),
        "inventory_prior_canonical_copy_rows": (
            inventory_payload.get("prior_canonical_copy_rows_before_dedup")
            if inventory_payload else None
        ),
        "dataset_record_count_after_dedup": int(dataset_payload["record_count"]),
        "dataset_independent_leakage_group_count": int(
            dataset_payload["independent_leakage_group_count"]
        ),
        "dataset_split_record_counts": dataset_payload["split_record_counts"],
        "dataset_rejected_counts": dataset_payload["rejected_counts"],
        "dataset_context_recovery": dataset_payload.get("context_recovery"),
        "dataset_adjudication": dataset_payload.get("adjudication"),
        "dataset_leakage_audit": dataset_payload.get("leakage_audit"),
        "dataset_manifest": str(dataset_manifest),
        "dataset_records": str(dataset_records),
        "dataset_adjudication_report": dataset_payload.get("adjudication_report"),
        "value_model_report": str(
            model_dir / "V26_HYDRAULIC_EXACT_RETURN_VALUE_MODEL_REPORT.json"
        ),
        "value_checkpoint": str(value_checkpoint),
        "benchmark_root": str(benchmark_dir),
        "old_roles_or_prior_versions_excluded": False,
        "step1_step2_prior_exposure_excluded": False,
        "offline_scientific_gate_between_training_and_benchmark": False,
        "baseline_rerun_requested": False,
        "new_counterfactual_swmm_truth_generated": False,
        "development_only": True,
        "ready_for_policy_lock": False,
    }
    summary_path = root / "V26_END_TO_END_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
