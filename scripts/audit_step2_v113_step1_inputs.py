"""Audit Step1 assets for an online-compatible V113 causal state input.

No Step1 training or SWMM execution occurs here.  If a previously generated
frozen-Step1 history report is supplied, it is imported as evidence with its
own provenance rather than silently presented as a current-head run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-manifest", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--train-index", required=True)
    ap.add_argument("--step1-checkpoint", required=True)
    ap.add_argument("--sensors", required=True)
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--history-report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = V60TrainCache(args.cache_manifest)
    fit, holdout = deterministic_rainfall_split_v60(cache, names=cache.names("D2"), holdout_fraction=0.20)
    fit_events = sorted({cache.entry(n).event_id for n in fit})
    holdout_events = sorted({cache.entry(n).event_id for n in holdout})
    eligible, ineligible = [], []
    for name in fit:
        entry = cache.entry(name)
        checkpoint_seconds = int(entry.arrays["elapsed_seconds"][entry.reference_index, 0])
        # The outer history ends at t.  Its earliest frame is t-3600 and
        # each frozen Step1 window needs a further 3600 s prefix, so the raw
        # compact must reach t-7200 (24 five-minute intervals), not 7800 s.
        if checkpoint_seconds >= 7200:
            eligible.append(name)
        else:
            ineligible.append(name)

    step1_meta_path = Path(args.step1_checkpoint).with_suffix(Path(args.step1_checkpoint).suffix + ".json")
    step1_meta = json.loads(step1_meta_path.read_text(encoding="utf-8")) if step1_meta_path.exists() else {}
    sensor_lines = [x.strip() for x in Path(args.sensors).read_text(encoding="utf-8").splitlines() if x.strip()]
    # Do not interpret unrelated JSON/CSV files as reconstructed trajectories.
    candidate_history_files = [
        str(p) for p in Path(args.study_root).rglob("*")
        if p.is_file() and any(token in p.name.lower() for token in ("reconstructed", "step1_history", "online_history"))
        and p.suffix.lower() in {".npz", ".npy", ".pt", ".json"}
    ]
    history_report = None
    if args.history_report and Path(args.history_report).exists():
        history_report = json.loads(Path(args.history_report).read_text(encoding="utf-8"))

    report = {
        "contract": "PROJECT7_STEP2_V113_STEP1_INPUT_AUDIT_V1",
        "git_head": _head(),
        "development_only": True,
        "swmm_run": False,
        "step1_retrained": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "lineage": {
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "checkpoint_table_sha256": _sha(args.checkpoints),
            "train_index_sha256": _sha(args.train_index),
            "step1_checkpoint_sha256": _sha(args.step1_checkpoint),
            "sensor_layout_sha256": _sha(args.sensors),
            "step1_checkpoint_meta": step1_meta,
        },
        "split": {
            "fit_groups": len(fit), "fit_events": fit_events,
            "holdout_groups_metadata_only": len(holdout), "holdout_events": holdout_events,
            "fit_holdout_event_overlap": sorted(set(fit_events) & set(holdout_events)),
        },
        "history_contract": {
            "frames": 13, "frame_seconds": 300, "window": "t-3600,...,t inclusive",
            "step1_reconstruction_requires_raw_prefix_seconds": 7200,
            "eligible_fit_groups": len(eligible),
            "ineligible_fit_groups": len(ineligible),
            "ineligible_reason": "checkpoint elapsed time < 7200 s; a full 13-frame frozen-Step1 reconstruction would require a pre-prefix before t-3600",
            "sensor_count": len(sensor_lines),
            "sensor_ids": sensor_lines,
            "future_leakage": "none by construction for frozen Step1 causal windows",
        },
        "online_compatible": {
            "reconstructed_history_cache_exists": False,
            "frozen_step1_checkpoint_loadable": bool(Path(args.step1_checkpoint).exists()),
            "eligible_group_coverage": f"{len(eligible)}/{len(fit)}",
            "production_ready": False,
            "requires_frozen_step1_inference": True,
        },
        "candidate_history_assets": candidate_history_files,
        "imported_prior_history_report": None,
    }
    if history_report is not None:
        report["imported_prior_history_report"] = {
            "path": str(Path(args.history_report).resolve()),
            "sha256": _sha(args.history_report),
            "git_head": history_report.get("lineage", {}).get("git_head"),
            "eligible_group_count": history_report.get("lineage", {}).get("eligible_d2_group_count"),
            "step1_reconstruction_vs_oracle_past": _json_safe(history_report.get("step1_reconstruction_vs_oracle_past")),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(Path(args.out)), "fit_groups": len(fit), "eligible": len(eligible), "online_cache": False}, indent=2))


if __name__ == "__main__":
    main()
