"""Run the complete Project7 V26 Development flow with one command.

Stages are deliberately simple and sequential:
  0. inventory reusable historical candidate-vs-HOLD exact-return JSONL when --study-root is used;
  1. consolidate exact-return truth and freeze Train/Validation/Test;
  2. train/select the V26 action-conditioned value model and report Test metrics;
  3. run all five Proposed Benchmark5 events while reusing immutable baselines.

No offline model-quality statistic can short-circuit Stage 3.  Only an actual program/data-lineage or
engineering-execution failure stops the workflow.
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


def _historical_sources(
    *,
    scripts: Path,
    root: Path,
    study_root: str | None,
    explicit_records: list[str],
) -> tuple[list[Path], Path | None]:
    """Return deduplicated candidate-truth sources, optionally discovered by the read-only inventory."""

    selected = [Path(value).resolve() for value in explicit_records]
    inventory_path: Path | None = None
    if study_root is not None:
        inventory_path = root / "V26_EXACT_RETURN_HISTORY_INVENTORY.json"
        _run(
            [
                sys.executable,
                str(scripts / "audit_project7_v26_exact_return_inventory.py"),
                "--root", str(Path(study_root).resolve()),
                "--out", str(inventory_path),
            ]
        )
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        reusable = inventory.get("reusable_files")
        if not isinstance(reusable, list):
            raise ValueError("V26 inventory lacks reusable_files")
        selected.extend(Path(str(value)).resolve() for value in reusable)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        deduped.append(path)
    if not deduped:
        raise ValueError("V26 requires --study-root and/or at least one --records-jsonl source")
    return deduped, inventory_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        help="Optional Project7 study tree to inventory automatically for reusable exact-return JSONL",
    )
    parser.add_argument("--records-jsonl", action="append", default=[])
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
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.out_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"V26 end-to-end output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    dataset_dir = root / "dataset"
    model_dir = root / "model"
    benchmark_dir = root / "benchmark5"

    records, inventory_path = _historical_sources(
        scripts=scripts,
        root=root,
        study_root=args.study_root,
        explicit_records=list(args.records_jsonl),
    )
    build = [
        sys.executable,
        str(scripts / "build_project7_v26_exact_return_dataset.py"),
    ]
    for path in records:
        build.extend(("--records-jsonl", str(path)))
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
    # Deliberately do not inspect AUC/sign/harmful-action metrics here. Benchmark5 is the next
    # scientific experiment, not a privilege granted by an offline gate.
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

    summary = {
        "contract": "PROJECT7_V26_END_TO_END_DEVELOPMENT_WORKFLOW_V2",
        "completed": True,
        "historical_inventory": str(inventory_path) if inventory_path is not None else None,
        "candidate_truth_source_count": len(records),
        "candidate_truth_sources": [str(path) for path in records],
        "dataset_manifest": str(dataset_manifest),
        "dataset_records": str(dataset_records),
        "value_model_report": str(model_dir / "V26_HYDRAULIC_EXACT_RETURN_VALUE_MODEL_REPORT.json"),
        "value_checkpoint": str(value_checkpoint),
        "benchmark_root": str(benchmark_dir),
        "offline_scientific_gate_between_training_and_benchmark": False,
        "baseline_rerun_requested": False,
        "development_only": True,
        "ready_for_policy_lock": False,
    }
    summary_path = root / "V26_END_TO_END_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
