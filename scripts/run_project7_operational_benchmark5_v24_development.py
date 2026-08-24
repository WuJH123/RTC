"""Run only V24 Proposed on the frozen five-event Development benchmark and reuse baselines.

The baseline cache, including Auto-RBC, is immutable.  This is the first required comparison for the
new RBC-informed stress-escape controller.  It does not access Validation or Final.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import run_project7_operational_benchmark5_current as base


def _run_proposed_v24(
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
            f"V24 Proposed result already exists for {event_id}; use a fresh --out-dir"
        )
    event_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v24_development.py"),
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
    if metadata.get("development_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("V24 operational run lost Development-only firewall")
    if metadata.get("historical_v23_evidence_mutated") is not False:
        raise RuntimeError("V24 metadata does not preserve historical V23 evidence")
    if metadata.get("high_stress_hold_escape_enabled") is not True:
        raise RuntimeError("V24 metadata did not enable the declared stress escape")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("V24 Proposed run used another event INP")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    original = base._run_proposed
    base._run_proposed = _run_proposed_v24
    try:
        base.main()
    finally:
        base._run_proposed = original


if __name__ == "__main__":
    main()
