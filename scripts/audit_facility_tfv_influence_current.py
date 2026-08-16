"""Build a rainfall-by-facility TFV influence map from existing authoritative trajectories.

Examples
--------
Analyze the current Project7 base cache plus D4 FIT/AUDIT caches::

    python scripts/audit_facility_tfv_influence_current.py \
      --cache base=.../CACHE_MANIFEST.json \
      --cache d4_fit=.../CACHE_MANIFEST.json \
      --cache d4_audit=.../CACHE_MANIFEST.json \
      --out-dir .../facility_tfv_influence

Inventory compatible historical caches under Project5-Project7 and restrict to selected storms::

    python scripts/audit_facility_tfv_influence_current.py \
      --discover-root E:/RTC_sewer/Project5 \
      --discover-root E:/RTC_sewer/Project6 \
      --discover-root E:/RTC_sewer/Project7/study_v069 \
      --rainfall T20_D120_chicago \
      --rainfall T100_* \
      --out-dir .../facility_tfv_influence_history

The audit is read-only, launches no SWMM, excludes Validation/Final/Formal-like splits by default,
and uses no surrogate prediction or gradient label for facility attribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.facility_tfv_influence import (
    CacheSpec,
    analyze_facility_tfv_influence,
    discover_compatible_cache_specs,
    parse_cache_spec,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Explicit V60-compatible CACHE_MANIFEST.json; repeatable.",
    )
    p.add_argument(
        "--discover-root",
        action="append",
        default=[],
        help="Read-only root scanned recursively for compatible CACHE_MANIFEST.json files.",
    )
    p.add_argument(
        "--rainfall",
        action="append",
        default=[],
        help="Rainfall-group glob to include (e.g. T20_D120_chicago or T100_*); repeatable.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--model-step-seconds", type=float, default=300.0)
    p.add_argument("--meaningful-absolute-m3", type=float, default=1.0)
    p.add_argument("--meaningful-relative", type=float, default=0.01)
    return p


def _deduplicate_specs(specs: list[CacheSpec]) -> list[CacheSpec]:
    seen: set[str] = set()
    result: list[CacheSpec] = []
    for spec in specs:
        path = str(Path(spec.manifest_path).resolve())
        if path in seen:
            continue
        seen.add(path)
        result.append(CacheSpec(label=spec.label, manifest_path=path))
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.model_step_seconds <= 0:
        raise ValueError("--model-step-seconds must be positive")
    if args.meaningful_absolute_m3 < 0 or args.meaningful_relative < 0:
        raise ValueError("meaningful thresholds must be non-negative")

    explicit = [parse_cache_spec(text) for text in args.cache]
    discovered, skipped = discover_compatible_cache_specs(args.discover_root)
    specs = _deduplicate_specs([*explicit, *discovered])
    if not specs:
        raise ValueError("no compatible caches supplied or discovered")

    exact, by_rain, global_facility, joint, report = analyze_facility_tfv_influence(
        specs,
        rainfall_patterns=tuple(args.rainfall),
        flood_rate_index=int(args.flood_rate_index),
        dt_seconds=float(args.model_step_seconds),
        meaningful_absolute_m3=float(args.meaningful_absolute_m3),
        meaningful_relative=float(args.meaningful_relative),
    )
    report["discovery"] = {
        "explicit_cache_count": len(explicit),
        "discovered_cache_count": len(discovered),
        "analyzed_cache_count": len(specs),
        "skipped": skipped,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exact_path = out / "FACILITY_TFV_EXACT_SINGLE_PAIR.csv"
    rain_path = out / "FACILITY_TFV_BY_RAINFALL.csv"
    global_path = out / "FACILITY_TFV_GLOBAL.csv"
    joint_path = out / "FACILITY_TFV_JOINT_CANDIDATES.csv"
    report_path = out / "FACILITY_TFV_INFLUENCE_REPORT.json"

    exact.to_csv(exact_path, index=False)
    by_rain.to_csv(rain_path, index=False)
    global_facility.to_csv(global_path, index=False)
    joint.to_csv(joint_path, index=False)
    report.update(
        {
            "outputs": {
                "exact_single_pair_csv": str(exact_path),
                "by_rainfall_csv": str(rain_path),
                "global_facility_csv": str(global_path),
                "joint_candidate_csv": str(joint_path),
            }
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
