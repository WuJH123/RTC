"""Run only Operational V23 on the frozen five-event benchmark and reuse the immutable baselines.

This wrapper deliberately reuses the comparison/aggregation code from
``run_project7_operational_benchmark5_current.py`` and swaps only the Proposed per-event runner.
No baseline cache is rebuilt.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import run_project7_operational_benchmark5_current as base


def _run_proposed_v23(
    *,
    event: dict[str, Any],
    asset_manifest: Path,
    v15_rank: Path,
    v21_boundary: Path,
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
        raise FileExistsError(
            f"V23 Proposed result already exists for {event_id}; use a fresh --out-dir"
        )
    event_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v23_development.py"),
        "--asset-manifest",
        str(asset_manifest),
        "--inp",
        str(event["inp_path"]),
        "--out-dir",
        str(event_root),
        "--run-id",
        run_id,
        "--v15-rank-checkpoint",
        str(v15_rank),
        "--v21-boundary-checkpoint",
        str(v21_boundary),
        "--device",
        device,
        "--decision-runtime-budget-seconds",
        str(float(budget)),
        "--probe-chunk-size",
        str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = base._metadata_and_stats(metadata_path)
    if metadata.get("operational_steering_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("V23 operational run lost Development-only firewall")
    if metadata.get("candidate_distribution_changed_after_v21_training") is not True:
        raise RuntimeError("V23 metadata did not disclose its changed candidate distribution")
    if metadata.get("candidate_generator_matches_v21_training") is not False:
        raise RuntimeError("V23 metadata falsely claims matched V21 candidate lineage")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("V23 Proposed run used another event INP")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    original = base._run_proposed
    base._run_proposed = _run_proposed_v23
    try:
        base.main()
    finally:
        base._run_proposed = original


if __name__ == "__main__":
    main()
