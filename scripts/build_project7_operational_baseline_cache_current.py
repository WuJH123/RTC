"""Compute the four fixed competitive baselines once for the frozen five-event benchmark.

Existing complete cache entries are validated and reused. Partial/stale entries fail closed rather
than being silently recomputed. All-open/all-closed are excluded because they are diagnostics, not
competitive comparators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.baseline_panel import baseline_lineage_failures
from rtc.operational_benchmark_v1 import (
    OPERATIONAL_BASELINE_CACHE_CONTRACT,
    OPERATIONAL_COMPARATORS,
    load_benchmark_manifest,
    sha256_file,
)
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from run_six_baselines_development_current import _run_one


def _load_cached_event(path: Path, *, event: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != OPERATIONAL_BASELINE_CACHE_CONTRACT:
        raise ValueError(f"wrong baseline cache contract: {path}")
    if str(payload.get("event_id")) != str(event["event_id"]):
        raise ValueError("baseline cache event ID mismatch")
    if str(payload.get("event_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise ValueError("baseline cache event INP changed")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline cache lacks rows")
    if tuple(str(row.get("strategy")) for row in rows) != OPERATIONAL_COMPARATORS:
        raise ValueError("baseline cache comparator order/set drifted")
    for row in rows:
        for key in ("metadata_path", "node_statistics_path"):
            if not Path(str(row.get(key, ""))).is_file():
                raise FileNotFoundError(f"cached baseline output missing: {row.get(key)}")
    return payload


def _build_event(
    *,
    event: dict[str, Any],
    sensors: Path,
    config: Path,
    native: Path,
    root: Path,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_root = root / event_id
    event_root.mkdir(parents=True, exist_ok=True)
    cache_path = event_root / "OPERATIONAL_FIXED_BASELINES.json"
    if cache_path.is_file():
        cached = _load_cached_event(cache_path, event=event)
        cached["cache_reused"] = True
        return cached

    # Refuse a half-existing cache: we never want an iteration to quietly mix old/new comparator runs.
    existing = [path for path in event_root.rglob("*.json") if path.name != cache_path.name]
    if existing:
        raise RuntimeError(
            f"partial baseline evidence exists for {event_id} but no complete cache manifest; "
            "audit/remove that dedicated cache directory before rebuilding"
        )

    rows = [
        _run_one(
            strategy=strategy,
            inp=Path(str(event["inp_path"])).resolve(),
            sensors=sensors,
            config=config,
            native_controls_template=native,
            root=event_root,
            event_id=event_id,
        )
        for strategy in OPERATIONAL_COMPARATORS
    ]
    reference = rows[0]["metadata"]
    assert isinstance(reference, dict)
    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        failures = baseline_lineage_failures(
            proposed=reference,
            baseline=metadata,
            expected_strategy=str(row["strategy"]),
        )
        if failures:
            raise RuntimeError(f"{event_id}/{row['strategy']} lineage mismatch: " + "; ".join(failures))
        row.pop("metadata")

    payload = {
        "contract": OPERATIONAL_BASELINE_CACHE_CONTRACT,
        "development_only": True,
        "event_id": event_id,
        "event_inp_path": str(Path(str(event["inp_path"])).resolve()),
        "event_inp_sha256": str(event["inp_sha256"]),
        "competitive_baselines": list(OPERATIONAL_COMPARATORS),
        "baseline_results_are_immutable_and_reused": True,
        "diagnostic_extremes_not_run": True,
        "cache_reused": False,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--native-controls-template", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    benchmark = load_benchmark_manifest(args.benchmark_manifest)
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    sensors = Path(practical_asset_path(assets, "sensors")).resolve()
    config = Path(practical_asset_path(assets, "config")).resolve()
    native = Path(args.native_controls_template).resolve()
    if not native.is_file():
        raise FileNotFoundError(native)
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    event_caches = [
        _build_event(event=event, sensors=sensors, config=config, native=native, root=root)
        for event in benchmark["events"]
    ]
    master = {
        "contract": OPERATIONAL_BASELINE_CACHE_CONTRACT,
        "development_only": True,
        "benchmark_manifest_path": str(Path(args.benchmark_manifest).resolve()),
        "benchmark_manifest_sha256": sha256_file(args.benchmark_manifest),
        "asset_manifest_sha256": sha256_file(args.asset_manifest),
        "native_controls_template_sha256": sha256_file(native),
        "event_count": 5,
        "competitive_baselines": list(OPERATIONAL_COMPARATORS),
        "baseline_results_are_immutable_and_reused": True,
        "events": [
            {
                "event_id": item["event_id"],
                "cache_path": str((root / str(item["event_id"]) / "OPERATIONAL_FIXED_BASELINES.json").resolve()),
                "cache_reused": bool(item["cache_reused"]),
            }
            for item in event_caches
        ],
    }
    out = root / "OPERATIONAL_BASELINE_CACHE5.json"
    out.write_text(json.dumps(master, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(master, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
