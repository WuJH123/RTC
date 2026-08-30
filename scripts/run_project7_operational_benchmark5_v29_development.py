"""Run Proposed V29 on frozen Operational Benchmark5 and reuse immutable baselines."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import run_project7_operational_benchmark5_current as base

V29_RUNTIME_CONTRACT = "PROJECT7_OPERATIONAL_DEVELOPMENT_V29_REGIME_BALANCED_MILD_CONTROL_V1"


def _run_proposed_v29(
    *,
    event: dict[str, Any],
    asset_manifest: Path,
    v15_rank: Path,
    v21_boundary: Path,
    v27_value: Path,
    v27_dataset_manifest: Path,
    v28r1_residual: Path,
    v29_value: Path,
    dataset_manifest: Path,
    root: Path,
    device: str,
    budget: float,
    probe_chunk: int,
) -> tuple[dict[str, Any], Path, Path]:
    event_id = str(event["event_id"])
    event_root = root / event_id
    run_id = f"{event_id}__proposed"
    metadata_path = event_root / f"{run_id}.json"
    if metadata_path.exists():
        raise FileExistsError(f"V29 Proposed result already exists for {event_id}")
    event_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v29_development.py"),
        "--asset-manifest", str(asset_manifest),
        "--inp", str(event["inp_path"]),
        "--out-dir", str(event_root),
        "--run-id", run_id,
        "--v15-rank-checkpoint", str(v15_rank),
        "--v21-boundary-checkpoint", str(v21_boundary),
        "--v27-value-checkpoint", str(v27_value),
        "--v27-dataset-manifest", str(v27_dataset_manifest),
        "--v28r1-residual-checkpoint", str(v28r1_residual),
        "--v29-value-checkpoint", str(v29_value),
        "--dataset-manifest", str(dataset_manifest),
        "--device", device,
        "--decision-runtime-budget-seconds", str(float(budget)),
        "--probe-chunk-size", str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = base._metadata_and_stats(metadata_path)
    if metadata.get("development_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("V29 operational run lost Development-only firewall")
    if metadata.get("v29_q95_mandatory") is not True:
        raise RuntimeError("V29 lost q95 execution")
    if metadata.get("v29_return_period_feature") is not False:
        raise RuntimeError("V29 illegally used return period as a policy input")
    if metadata.get("v29_event_duration_feature") is not False:
        raise RuntimeError("V29 illegally used event duration as a policy input")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("V29 Proposed run used another event INP")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument("--v27-dataset-manifest", required=True)
    parser.add_argument("--v28r1-residual-checkpoint", required=True)
    parser.add_argument("--v29-value-checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()

    original = base._run_proposed
    base._run_proposed = lambda **kwargs: _run_proposed_v29(
        **kwargs,
        v27_value=Path(args.v27_value_checkpoint).resolve(),
        v27_dataset_manifest=Path(args.v27_dataset_manifest).resolve(),
        v28r1_residual=Path(args.v28r1_residual_checkpoint).resolve(),
        v29_value=Path(args.v29_value_checkpoint).resolve(),
        dataset_manifest=Path(args.dataset_manifest).resolve(),
    )
    original_argv = list(sys.argv)
    try:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--benchmark-manifest", str(Path(args.benchmark_manifest).resolve()),
            "--baseline-cache", str(Path(args.baseline_cache).resolve()),
            "--asset-manifest", str(Path(args.asset_manifest).resolve()),
            "--priority-nodes", str(Path(args.priority_nodes).resolve()),
            "--v15-rank-checkpoint", str(Path(args.v15_rank_checkpoint).resolve()),
            "--v21-boundary-checkpoint", str(Path(args.v21_boundary_checkpoint).resolve()),
            "--out-dir", str(Path(args.out_dir).resolve()),
            "--device", str(args.device),
            "--decision-runtime-budget-seconds", str(float(args.decision_runtime_budget_seconds)),
            "--probe-chunk-size", str(int(args.probe_chunk_size)),
        ]
        base.main()
    finally:
        sys.argv = original_argv
        base._run_proposed = original


if __name__ == "__main__":
    main()
