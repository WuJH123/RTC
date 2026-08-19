from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from rtc.baseline_panel import (
    SCIENTIFIC_COMPARATOR_IDS,
    build_direct_tfv_baseline_comparison,
    inspect_baseline_artifact,
)
from rtc.baselines import FORMAL_FIXED_BASELINE_IDS
from rtc.runtime_failure_diagnostics import summarize_runtime_failures


def _write_stats(path: Path, value: float) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "delta_flooding_volume_m3"])
        writer.writeheader()
        writer.writerow({"node_id": "N1", "delta_flooding_volume_m3": value})


def _write_metadata(root: Path, strategy: str, tfv: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stats = root / f"{strategy}.node_statistics.csv.gz"
    _write_stats(stats, tfv)
    payload = {
        "run_id": f"event__{strategy}",
        "strategy": strategy,
        "source_inp_sha256": "inp",
        "controller_config_sha256": "cfg",
        "swmm_engine_version": "5.2.4",
        "prepared_event_clock": {"simulation_start": "2026-01-01T00:00:00"},
        "node_statistics_file": stats.name,
        "global_peak_flood_rate_m3s": 1.0,
        "flow_routing_error_pct": 0.1,
    }
    path = root / f"{strategy}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_legacy_baseline_without_lineage_requires_fresh_run(tmp_path: Path) -> None:
    proposed = _write_metadata(tmp_path / "proposed", "proposed_direct_tfv_all109_receding_mpc", 80.0)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"strategy": "no_control"}), encoding="utf-8")
    result = inspect_baseline_artifact(
        proposed_metadata=proposed,
        baseline_metadata=legacy,
        expected_strategy="no_control",
    )
    assert result["verified"] is False
    assert result["reuse_allowed"] is False
    assert result["fresh_run_required"] is True
    assert any("source_inp_sha256" in text for text in result["failures"])


def test_clean_proposed_is_compared_against_all_six_verified_baselines(tmp_path: Path) -> None:
    proposed = _write_metadata(tmp_path / "proposed", "proposed_direct_tfv_all109_receding_mpc", 80.0)
    tfv = {
        "no_control": 100.0,
        "internal_rtc": 90.0,
        "auto_rbc": 95.0,
        "efd": 85.0,
        "all_open": 120.0,
        "all_closed": 130.0,
    }
    baseline_paths = {
        strategy: _write_metadata(tmp_path / strategy, strategy, tfv[strategy])
        for strategy in FORMAL_FIXED_BASELINE_IDS
    }
    payload = build_direct_tfv_baseline_comparison(
        proposed_metadata=proposed,
        baseline_metadata_by_strategy=baseline_paths,
    )
    assert payload["baseline_provenance_verified_all"] is True
    assert payload["proposed_beats_no_control"] is True
    assert payload["classification"] == "DEVELOPMENT_EVENT_METHOD_ADVANTAGE_SUPPORTED"
    wins = payload["proposed_beats_operational_comparator"]
    assert set(wins) == set(SCIENTIFIC_COMPARATOR_IDS)
    assert all(wins.values())
    assert payload["diagnostic_extremes"] == ["all_open", "all_closed"]
    assert payload["universal_comparator_superiority_required"] is False
    rows = payload["rows"]
    assert isinstance(rows, list) and len(rows) == 7
    role_by_strategy = {row["strategy"]: row["role"] for row in rows}
    assert role_by_strategy["all_open"] == "diagnostic_extreme"
    assert role_by_strategy["all_closed"] == "diagnostic_extreme"


def test_accelerator_fallback_is_not_mislabeled_as_scientific_policy_failure() -> None:
    rows = [
        {
            "elapsed_seconds": 31800 + index * 600,
            "source": "FALLBACK_RUNTIME_ERROR",
            "diagnostics": {
                "error_type": "AcceleratorError",
                "error": "CUDA error: unknown error",
            },
        }
        for index in range(7)
    ]
    payload = summarize_runtime_failures(rows)
    assert payload["fallback_count"] == 7
    assert payload["counts_by_domain"]["accelerator_environment"] == 7
    assert payload["classification"] == "ACCELERATOR_ENVIRONMENT_ONLY"
    assert payload["scientific_policy_failure_inferred_from_accelerator_error"] is False


def test_non_cuda_runtime_error_remains_controller_runtime_failure() -> None:
    payload = summarize_runtime_failures(
        [
            {
                "elapsed_seconds": 600,
                "source": "FALLBACK_RUNTIME_ERROR",
                "diagnostics": {"error_type": "ValueError", "error": "bad tensor shape"},
            }
        ]
    )
    assert payload["counts_by_domain"]["controller_runtime"] == 1
    assert payload["classification"] == "CONTROLLER_RUNTIME_FAILURE_PRESENT"
