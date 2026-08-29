"""Run the V28 Development residual training and one Proposed Benchmark5 lane.

The wrapper intentionally does not create a dataset or truth.  It consumes an already audited
V27 exact-return dataset, fits V28, and then runs only Proposed V28 while the immutable baseline
cache supplies the comparison strategies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument(
        "--v27-dataset-manifest",
        help="Immutable V27 dataset manifest used only for frozen Q27 lineage validation",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-records", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--truth-plan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"V28 end-to-end output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir = out_root / "model"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_project7_v28_q95_residual_value.py"),
        "--asset-manifest", str(Path(args.asset_manifest).resolve()),
        "--v27-value-checkpoint", str(Path(args.v27_value_checkpoint).resolve()),
        "--v27-dataset-manifest", str(
            Path(args.v27_dataset_manifest or args.dataset_manifest).resolve()
        ),
        "--dataset-manifest", str(Path(args.dataset_manifest).resolve()),
        "--dataset-records", str(Path(args.dataset_records).resolve()),
        "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
        "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
        "--out-dir", str(model_dir),
        "--study-root", str(Path(args.study_root).resolve()),
        "--device", str(args.device),
        "--probe-chunk-size", str(int(args.probe_chunk_size)),
        "--cv-folds", str(int(args.cv_folds)),
    ]
    if args.truth_plan:
        command.extend(["--truth-plan", str(Path(args.truth_plan).resolve())])
    subprocess.run(command, check=True)
    benchmark_dir = out_root / "benchmark5"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_project7_operational_benchmark5_v28_development.py"),
        "--benchmark-manifest", str(Path(args.benchmark_manifest).resolve()),
        "--baseline-cache", str(Path(args.baseline_cache).resolve()),
        "--asset-manifest", str(Path(args.asset_manifest).resolve()),
        "--priority-nodes", str(Path(args.priority_nodes).resolve()),
        "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
        "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
        "--v27-value-checkpoint", str(Path(args.v27_value_checkpoint).resolve()),
        "--v27-dataset-manifest", str(
            Path(args.v27_dataset_manifest or args.dataset_manifest).resolve()
        ),
        "--v28-residual-checkpoint", str(model_dir / "V28_Q95_MATCHED_RESIDUAL_VALUE_MODEL.pt"),
        "--dataset-manifest", str(Path(args.dataset_manifest).resolve()),
        "--out-dir", str(benchmark_dir),
        "--device", str(args.device),
        "--decision-runtime-budget-seconds", str(float(args.decision_runtime_budget_seconds)),
        "--probe-chunk-size", str(int(args.probe_chunk_size)),
    ]
    subprocess.run(command, check=True)
    print(f"V28 model output: {model_dir}")
    print(f"V28 Benchmark5 output: {benchmark_dir}")


if __name__ == "__main__":
    main()
