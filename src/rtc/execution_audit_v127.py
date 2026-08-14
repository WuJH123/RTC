"""Post-run execution audits for Project7 V127 and fixed Python comparators."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

V127_TARGET_WRITE_AUDIT_CONTRACT = "PROJECT7_TARGET_WRITE_SAME_EPOCH_READBACK_V1"


def audit_target_write_readback_v127(
    *,
    metadata_path: str | Path,
    tolerance: float = 1.0e-6,
) -> dict[str, object]:
    """Compare every logged command with compact target_setting at the same SWMM epoch.

    ``run_authoritative_closed_loop`` writes commands and then records the compact tensor at
    that same elapsed time.  This audit therefore checks target-latch acceptance without
    inserting another PySWMM write/read cycle into the hydraulic simulation.  Realised
    ``current_setting`` is intentionally not part of this write-acceptance test because it
    may physically lag the supervisory target.
    """
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root = meta_path.parent
    compact_path = root / str(meta["compact_file"])
    decision_path = root / str(meta["decision_file"])
    if not compact_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("V127 execution audit cannot locate compact/decision artifact")

    with np.load(compact_path, allow_pickle=False) as raw:
        elapsed = np.asarray(raw["elapsed_seconds"], dtype=np.int64)
        actuator_ids = tuple(raw["actuator_ids"].astype(str).tolist())
        target = np.asarray(raw["target_setting"], dtype=np.float64)
    if target.shape != (len(elapsed), len(actuator_ids)):
        raise ValueError("compact target_setting tensor shape is invalid")
    if len(set(elapsed.tolist())) != len(elapsed):
        raise ValueError("compact elapsed_seconds are not unique")
    time_index = {int(t): i for i, t in enumerate(elapsed.tolist())}

    decisions = [
        json.loads(line)
        for line in decision_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_error = 0.0
    failed: list[dict[str, object]] = []
    for item in decisions:
        when = int(item["elapsed_seconds"])
        index = time_index.get(when)
        if index is None:
            failed.append({"elapsed_seconds": when, "reason": "missing_same_epoch_compact_row"})
            continue
        settings = item.get("settings")
        if not isinstance(settings, dict) or set(settings) != set(actuator_ids):
            failed.append({"elapsed_seconds": when, "reason": "incomplete_decision_setting_vector"})
            continue
        requested = np.asarray([float(settings[aid]) for aid in actuator_ids], dtype=np.float64)
        if not np.isfinite(requested).all():
            failed.append({"elapsed_seconds": when, "reason": "nonfinite_requested_setting"})
            continue
        error = float(np.max(np.abs(requested - target[index]), initial=0.0))
        max_error = max(max_error, error)
        if error > float(tolerance):
            failed.append(
                {
                    "elapsed_seconds": when,
                    "reason": "target_write_readback_mismatch",
                    "max_error": error,
                }
            )
    return {
        "contract": V127_TARGET_WRITE_AUDIT_CONTRACT,
        "decision_count": len(decisions),
        "max_target_write_readback_error": max_error,
        "tolerance": float(tolerance),
        "failed_decisions": len(failed),
        "passed": len(failed) == 0,
        "failures": failed[:20],
        "current_setting_used_as_write_acceptance": False,
    }


__all__ = ["V127_TARGET_WRITE_AUDIT_CONTRACT", "audit_target_write_readback_v127"]
