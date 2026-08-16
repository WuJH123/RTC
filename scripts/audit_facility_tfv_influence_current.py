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
and uses no surrogate prediction or gradient label for facility attribution. Historical caches are
merged only when they have the same ordered 109-actuator and node-ID signature; incompatible
Project generations are reported and excluded rather than silently mixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.facility_tfv_influence import (
    CacheSpec,
    analyze_facility_tfv_influence,
    discover_compatible_cache_specs,
    parse_cache_spec,
)
from rtc.step2_train_response_v60 import V60TrainCache


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
    p.add_argument("--expected-actuators", type=int, default=109)
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


def _id_vector(arrays: dict[str, np.ndarray], name: str, reference: int, width: int | None = None) -> list[str]:
    raw = np.asarray(arrays[name])
    if raw.ndim == 1 and (width is None or raw.size == width):
        values = raw.reshape(-1)
    else:
        values = np.asarray(raw[reference]).reshape(-1)
    if width is not None and values.size != int(width):
        raise ValueError(f"{name} count {values.size} differs from expected {width}")
    return [str(value) for value in values]


def _asset_signature(spec: CacheSpec) -> dict[str, object]:
    cache = V60TrainCache(spec.manifest_path)
    names = cache.names()
    if not names:
        raise ValueError(f"empty cache: {spec.manifest_path}")
    entry = cache.entry(names[0])
    arrays = entry.arrays
    ref = int(entry.reference_index)
    settings = np.asarray(arrays["settings"][ref])
    if settings.ndim != 2:
        raise ValueError(f"invalid settings shape in {spec.manifest_path}")
    actuator_ids = _id_vector(arrays, "actuator_ids", ref, int(settings.shape[-1]))
    node_ids = _id_vector(arrays, "node_ids", ref) if "node_ids" in arrays else []
    canonical = json.dumps(
        {"actuator_ids": actuator_ids, "node_ids": node_ids},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return {
        "actuator_count": len(actuator_ids),
        "actuator_ids": actuator_ids,
        "node_count": len(node_ids),
        "asset_signature_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _filter_asset_lineage(
    specs: list[CacheSpec], *, expected_actuators: int
) -> tuple[list[CacheSpec], list[str], list[dict[str, object]], str]:
    if expected_actuators <= 0:
        raise ValueError("--expected-actuators must be positive")
    compatible: list[CacheSpec] = []
    excluded: list[dict[str, object]] = []
    reference_signature = ""
    universe: list[str] = []
    for spec in specs:
        try:
            signature = _asset_signature(spec)
        except Exception as exc:
            excluded.append(
                {
                    "label": spec.label,
                    "manifest_path": spec.manifest_path,
                    "reason": f"asset_signature_error:{type(exc).__name__}:{exc}",
                }
            )
            continue
        if int(signature["actuator_count"]) != int(expected_actuators):
            excluded.append(
                {
                    "label": spec.label,
                    "manifest_path": spec.manifest_path,
                    "reason": "actuator_count_mismatch",
                    **{key: value for key, value in signature.items() if key != "actuator_ids"},
                }
            )
            continue
        current = str(signature["asset_signature_sha256"])
        if not reference_signature:
            reference_signature = current
            universe = list(signature["actuator_ids"])
        if current != reference_signature:
            excluded.append(
                {
                    "label": spec.label,
                    "manifest_path": spec.manifest_path,
                    "reason": "asset_signature_mismatch",
                    **{key: value for key, value in signature.items() if key != "actuator_ids"},
                }
            )
            continue
        compatible.append(spec)
    if not compatible:
        raise ValueError(
            f"no caches match the required {expected_actuators}-actuator physical asset signature"
        )
    return compatible, universe, excluded, reference_signature


def _complete_facility_tables(
    by_rain: pd.DataFrame,
    global_facility: pd.DataFrame,
    *,
    rainfalls: list[str],
    actuator_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep untested facilities visible instead of silently treating missing evidence as zero effect."""
    if not rainfalls:
        return by_rain, global_facility
    rain_grid = pd.MultiIndex.from_product(
        [rainfalls, actuator_ids], names=["rainfall_group", "actuator_id"]
    ).to_frame(index=False)
    rain = rain_grid.merge(by_rain, on=["rainfall_group", "actuator_id"], how="left")
    global_grid = pd.DataFrame({"actuator_id": actuator_ids})
    global_out = global_grid.merge(global_facility, on="actuator_id", how="left")

    count_columns = [
        "tested_pairs",
        "tested_checkpoints",
        "tested_events",
        "beneficial_pairs",
        "harmful_pairs",
        "below_threshold_pairs",
        "delayed_beneficial_pairs",
    ]
    fraction_columns = ["beneficial_fraction", "harmful_fraction"]
    for frame in (rain, global_out):
        for column in count_columns:
            if column in frame:
                frame[column] = frame[column].fillna(0).astype(int)
        for column in fraction_columns:
            if column in frame:
                frame[column] = frame[column].fillna(0.0).astype(float)
        if "evidence_class" in frame:
            frame["evidence_class"] = frame["evidence_class"].fillna("UNTESTED_SINGLE_ACTUATOR")
        if "has_sampled_control_value" in frame:
            frame["has_sampled_control_value"] = frame["has_sampled_control_value"].fillna(False).astype(bool)
    return rain, global_out


def main() -> None:
    args = _parser().parse_args()
    if args.model_step_seconds <= 0:
        raise ValueError("--model-step-seconds must be positive")
    if args.meaningful_absolute_m3 < 0 or args.meaningful_relative < 0:
        raise ValueError("meaningful thresholds must be non-negative")

    explicit = [parse_cache_spec(text) for text in args.cache]
    discovered, skipped = discover_compatible_cache_specs(args.discover_root)
    candidate_specs = _deduplicate_specs([*explicit, *discovered])
    if not candidate_specs:
        raise ValueError("no compatible caches supplied or discovered")
    specs, actuator_ids, lineage_excluded, asset_signature = _filter_asset_lineage(
        candidate_specs, expected_actuators=int(args.expected_actuators)
    )

    exact, by_rain, global_facility, joint, report = analyze_facility_tfv_influence(
        specs,
        rainfall_patterns=tuple(args.rainfall),
        flood_rate_index=int(args.flood_rate_index),
        dt_seconds=float(args.model_step_seconds),
        meaningful_absolute_m3=float(args.meaningful_absolute_m3),
        meaningful_relative=float(args.meaningful_relative),
    )
    rainfalls = [str(value) for value in report.get("rainfall_groups_analyzed", [])]
    by_rain, global_facility = _complete_facility_tables(
        by_rain,
        global_facility,
        rainfalls=rainfalls,
        actuator_ids=actuator_ids,
    )
    report.update(
        {
            "actuator_ids_seen": len(actuator_ids),
            "asset_signature_sha256": asset_signature,
            "rainfall_actuator_matrix_rows": int(len(by_rain)),
            "rainfall_actuator_cells_tested": int((by_rain["tested_pairs"] > 0).sum()) if not by_rain.empty else 0,
            "rainfall_actuator_cells_untested": int((by_rain["tested_pairs"] == 0).sum()) if not by_rain.empty else 0,
            "discovery": {
                "explicit_cache_count": len(explicit),
                "discovered_cache_count": len(discovered),
                "candidate_cache_count": len(candidate_specs),
                "analyzed_cache_count": len(specs),
                "skipped_incompatible_cache_contract": skipped,
                "excluded_physical_asset_lineage": lineage_excluded,
            },
        }
    )

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
    report["outputs"] = {
        "exact_single_pair_csv": str(exact_path),
        "by_rainfall_csv": str(rain_path),
        "global_facility_csv": str(global_path),
        "joint_candidate_csv": str(joint_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
