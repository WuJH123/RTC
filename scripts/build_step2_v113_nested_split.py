"""Freeze the forcing-only nested development split for V11.3."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v113_audit import deterministic_event_split_v113


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--atlas", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = V60TrainCache(args.cache_manifest)
    d2 = cache.names("D2")
    fit, holdout = deterministic_rainfall_split_v60(cache, names=d2, holdout_fraction=0.20)
    fit_events = sorted({cache.entry(name).event_id for name in fit})
    nested = deterministic_event_split_v113(fit_events, devfit_count=10)
    devfit_events = set(nested["devfit_events"])
    devcheck_events = set(nested["devcheck_events"])
    devfit = [name for name in fit if cache.entry(name).event_id in devfit_events]
    devcheck = [name for name in fit if cache.entry(name).event_id in devcheck_events]
    if not devfit or not devcheck or devfit_events & devcheck_events:
        raise RuntimeError("V113 nested split is empty or event-overlapping")
    devfit_rain = {cache.entry(name).rainfall_group for name in devfit}
    devcheck_rain = {cache.entry(name).rainfall_group for name in devcheck}
    if devfit_rain & devcheck_rain:
        raise RuntimeError("V113 nested split has rainfall-group overlap")

    def side(names: list[str]) -> dict[str, object]:
        return {
            "groups": len(names),
            # Keep the exact cache group keys so downstream DevFit-only
            # builders cannot accidentally reconstruct the split by outcome.
            "group_names": sorted(names),
            "events": sorted({cache.entry(name).event_id for name in names}),
            "rainfall_groups": sorted({cache.entry(name).rainfall_group for name in names}),
            "group_digest": _digest(names),
            "event_digest": _digest(sorted({cache.entry(name).event_id for name in names})),
            "rainfall_digest": _digest(sorted({cache.entry(name).rainfall_group for name in names})),
        }

    report = {
        "contract": "PROJECT7_STEP2_V113_NESTED_FORCING_SPLIT_V1",
        "git_head": _git_head(),
        "development_only": True,
        "new_swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "outcomes_used_for_selection": False,
        "split_contract": {
            "original_split": "deterministic_rainfall_split_v60 holdout_fraction=0.20",
            "nested_selection": nested,
            "devfit_event_count": len(devfit_events),
            "devcheck_event_count": len(devcheck_events),
            "event_overlap": sorted(devfit_events & devcheck_events),
            "rainfall_overlap": sorted(devfit_rain & devcheck_rain),
        },
        "original_train_internal_split": {"fit": side(fit), "holdout_metadata_only": side(holdout)},
        "v113_devfit": side(devfit),
        "v113_devcheck": side(devcheck),
        "prior_source_events": sorted(devfit_events),
        "normalization_source_events": sorted(devfit_events),
        "threshold_source_events": sorted(devfit_events),
        "lineage": {
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "graph_sha256": _sha256(args.graph),
            "atlas_sha256": _sha256(args.atlas) if args.atlas else None,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
