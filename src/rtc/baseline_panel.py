from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from .baselines import (
    COMPETITIVE_BASELINE_IDS,
    FORMAL_FIXED_BASELINE_IDS,
    canonical_baseline_id,
)


CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT = (
    "PROJECT7_CURRENT_SIX_FIXED_BASELINE_DEVELOPMENT_SWMM_V1"
)
DIRECT_TFV_BASELINE_PANEL_CONTRACT = "PROJECT7_DIRECT_TFV_DEVELOPMENT_BASELINE_PANEL_V2_ROLE_AWARE"
DIRECT_TFV_BASELINE_PROVENANCE_CONTRACT = "PROJECT7_FIXED_BASELINE_PROVENANCE_V1"
SCIENTIFIC_COMPARATOR_IDS = ("internal_rtc", "auto_rbc", "efd")
DIAGNOSTIC_EXTREME_IDS = ("all_open", "all_closed")
_REQUIRED_LINEAGE_FIELDS = (
    "source_inp_sha256",
    "controller_config_sha256",
    "swmm_engine_version",
    "prepared_event_clock",
)


def _json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return payload


def _statistics_path(metadata_path: Path, metadata: Mapping[str, Any]) -> Path:
    name = metadata.get("node_statistics_file")
    if not name:
        raise ValueError(f"metadata lacks node_statistics_file: {metadata_path}")
    path = metadata_path.parent / str(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def tfv_m3(statistics_path: str | Path) -> float:
    total = 0.0
    with gzip.open(statistics_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += float(row["delta_flooding_volume_m3"])
    return float(total)


def baseline_lineage_failures(
    *,
    proposed: Mapping[str, Any],
    baseline: Mapping[str, Any],
    expected_strategy: str,
) -> list[str]:
    """Return fail-closed lineage mismatches without inventing missing legacy metadata."""
    expected = canonical_baseline_id(expected_strategy)
    failures: list[str] = []
    actual_strategy = canonical_baseline_id(str(baseline.get("strategy", "")))
    if actual_strategy != expected:
        failures.append(f"strategy={baseline.get('strategy')!r}, expected {expected!r}")
    for key in _REQUIRED_LINEAGE_FIELDS:
        left = proposed.get(key)
        right = baseline.get(key)
        if left in (None, ""):
            failures.append(f"proposed metadata missing lineage field {key}")
        if right in (None, ""):
            failures.append(f"baseline metadata missing lineage field {key}")
            continue
        if left not in (None, "") and left != right:
            failures.append(f"{key} mismatch")
    return failures


def inspect_baseline_artifact(
    *,
    proposed_metadata: str | Path,
    baseline_metadata: str | Path,
    expected_strategy: str,
) -> dict[str, Any]:
    """Classify a baseline artifact as reusable or requiring a fresh authoritative run."""
    proposed_path = Path(proposed_metadata).resolve()
    baseline_path = Path(baseline_metadata).resolve()
    if not baseline_path.is_file():
        return {
            "contract": DIRECT_TFV_BASELINE_PROVENANCE_CONTRACT,
            "strategy": canonical_baseline_id(expected_strategy),
            "metadata_path": str(baseline_path),
            "node_statistics_path": None,
            "verified": False,
            "reuse_allowed": False,
            "fresh_run_required": True,
            "failures": ["baseline metadata missing"],
        }
    proposed = _json_object(proposed_path)
    baseline = _json_object(baseline_path)
    failures = baseline_lineage_failures(
        proposed=proposed,
        baseline=baseline,
        expected_strategy=expected_strategy,
    )
    statistics_path: Path | None = None
    try:
        statistics_path = _statistics_path(baseline_path, baseline)
    except (ValueError, FileNotFoundError) as exc:
        failures.append(str(exc))
    verified = not failures
    return {
        "contract": DIRECT_TFV_BASELINE_PROVENANCE_CONTRACT,
        "strategy": canonical_baseline_id(expected_strategy),
        "metadata_path": str(baseline_path),
        "node_statistics_path": None if statistics_path is None else str(statistics_path.resolve()),
        "verified": verified,
        "reuse_allowed": verified,
        "fresh_run_required": not verified,
        "failures": failures,
    }


def _role(strategy: str) -> str:
    if strategy == "proposed":
        return "proposed"
    if strategy == "no_control":
        return "primary_reference"
    if strategy in SCIENTIFIC_COMPARATOR_IDS:
        return "operational_comparator"
    return "diagnostic_extreme"


def build_direct_tfv_baseline_comparison(
    *,
    proposed_metadata: str | Path,
    baseline_metadata_by_strategy: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build one event comparison after all six fixed evidence baselines pass provenance checks."""
    proposed_path = Path(proposed_metadata).resolve()
    proposed = _json_object(proposed_path)
    proposed_statistics = _statistics_path(proposed_path, proposed)
    expected = tuple(FORMAL_FIXED_BASELINE_IDS)
    normalized = {
        canonical_baseline_id(key): value for key, value in baseline_metadata_by_strategy.items()
    }
    if set(normalized) != set(expected):
        raise ValueError(
            f"comparison requires exactly the six fixed evidence baselines; "
            f"missing={sorted(set(expected)-set(normalized))}, "
            f"extra={sorted(set(normalized)-set(expected))}"
        )

    proposed_tfv = tfv_m3(proposed_statistics)
    rows: list[dict[str, Any]] = [
        {
            "strategy": "proposed",
            "role": _role("proposed"),
            "tfv_m3": proposed_tfv,
            "global_peak_flood_rate_m3s": float(proposed.get("global_peak_flood_rate_m3s", 0.0)),
            "flow_routing_error_pct": float(proposed.get("flow_routing_error_pct", 0.0)),
            "metadata_path": str(proposed_path),
            "node_statistics_path": str(proposed_statistics.resolve()),
            "provenance_verified": True,
        }
    ]
    for strategy in expected:
        metadata_path = Path(normalized[strategy]).resolve()
        baseline = _json_object(metadata_path)
        failures = baseline_lineage_failures(
            proposed=proposed,
            baseline=baseline,
            expected_strategy=strategy,
        )
        if failures:
            raise ValueError(f"{strategy} baseline provenance mismatch: " + "; ".join(failures))
        statistics_path = _statistics_path(metadata_path, baseline)
        rows.append(
            {
                "strategy": strategy,
                "role": _role(strategy),
                "tfv_m3": tfv_m3(statistics_path),
                "global_peak_flood_rate_m3s": float(
                    baseline.get("global_peak_flood_rate_m3s", 0.0)
                ),
                "flow_routing_error_pct": float(baseline.get("flow_routing_error_pct", 0.0)),
                "metadata_path": str(metadata_path),
                "node_statistics_path": str(statistics_path.resolve()),
                "provenance_verified": True,
            }
        )

    by_strategy = {str(row["strategy"]): row for row in rows}
    no_control_tfv = float(by_strategy["no_control"]["tfv_m3"])
    for row in rows:
        tfv = float(row["tfv_m3"])
        row["delta_tfv_vs_no_control_m3"] = tfv - no_control_tfv
        row["tfv_reduction_vs_no_control_pct"] = (
            100.0 * (no_control_tfv - tfv) / no_control_tfv if no_control_tfv > 0.0 else None
        )
        if row["strategy"] != "proposed":
            row["proposed_minus_strategy_tfv_m3"] = proposed_tfv - tfv
            row["proposed_reduction_vs_strategy_pct"] = (
                100.0 * (tfv - proposed_tfv) / tfv if tfv > 0.0 else None
            )

    proposed_beats_no_control = proposed_tfv < no_control_tfv
    comparator_wins = {
        strategy: proposed_tfv < float(by_strategy[strategy]["tfv_m3"])
        for strategy in SCIENTIFIC_COMPARATOR_IDS
    }
    if proposed_beats_no_control and all(comparator_wins.values()):
        classification = "DEVELOPMENT_EVENT_METHOD_ADVANTAGE_SUPPORTED"
    elif proposed_beats_no_control:
        classification = "USEFUL_VS_NO_CONTROL_COMPARATOR_SUPERIORITY_NOT_UNIVERSAL"
    else:
        classification = "NO_CONTROL_BENEFIT_NOT_SUPPORTED"

    return {
        "contract": DIRECT_TFV_BASELINE_PANEL_CONTRACT,
        "development_only": True,
        "event_run_id": proposed.get("run_id"),
        "proposed_metadata": str(proposed_path),
        "baseline_provenance_verified_all": True,
        "competitive_baselines": list(COMPETITIVE_BASELINE_IDS),
        "operational_comparators": list(SCIENTIFIC_COMPARATOR_IDS),
        "diagnostic_extremes": list(DIAGNOSTIC_EXTREME_IDS),
        "universal_comparator_superiority_required": False,
        "global_peak_role": "report_only",
        "rows": rows,
        "proposed_beats_no_control": proposed_beats_no_control,
        "proposed_beats_operational_comparator": comparator_wins,
        "classification": classification,
    }


__all__ = [
    "CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT",
    "DIAGNOSTIC_EXTREME_IDS",
    "DIRECT_TFV_BASELINE_PANEL_CONTRACT",
    "DIRECT_TFV_BASELINE_PROVENANCE_CONTRACT",
    "SCIENTIFIC_COMPARATOR_IDS",
    "baseline_lineage_failures",
    "build_direct_tfv_baseline_comparison",
    "inspect_baseline_artifact",
    "tfv_m3",
]
