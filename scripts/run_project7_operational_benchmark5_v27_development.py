"""Run Proposed V27 on frozen Operational Benchmark5 and reuse immutable baselines."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import run_project7_operational_benchmark5_current as base


def _run_proposed_v27(
    *,
    event: dict[str, Any],
    asset_manifest: Path,
    v15_rank: Path,
    v21_boundary: Path,
    v27_value: Path,
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
        raise FileExistsError(f"V27 Proposed result already exists for {event_id}; use a fresh --out-dir")
    event_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v27_development.py"),
        "--asset-manifest", str(asset_manifest),
        "--inp", str(event["inp_path"]),
        "--out-dir", str(event_root),
        "--run-id", run_id,
        "--v15-rank-checkpoint", str(v15_rank),
        "--v21-boundary-checkpoint", str(v21_boundary),
        "--v27-value-checkpoint", str(v27_value),
        "--dataset-manifest", str(dataset_manifest),
        "--device", device,
        "--decision-runtime-budget-seconds", str(float(budget)),
        "--probe-chunk-size", str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = base._metadata_and_stats(metadata_path)
    if metadata.get("development_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("V27 operational run lost Development-only firewall")
    if metadata.get("scientific_metrics_block_runtime") is not False:
        raise RuntimeError("V27 unexpectedly restored an offline scientific runtime gate")
    if metadata.get("v27_auto_rbc_shadow_is_candidate_only") is not True:
        raise RuntimeError("V27 Auto-RBC shadow is not marked candidate-only")
    if metadata.get("v27_runtime_ranking_uses_unclipped_latent") is not True:
        raise RuntimeError("V27 runtime lost unclipped-latent ranking")
    if metadata.get("v27_q95_support_execution") is not True:
        raise RuntimeError("V27 primary execution lost q95 support")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("V27 Proposed run used another event INP")
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
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()

    original = base._run_proposed
    base._run_proposed = lambda **kwargs: _run_proposed_v27(
        **kwargs,
        v27_value=Path(args.v27_value_checkpoint).resolve(),
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
