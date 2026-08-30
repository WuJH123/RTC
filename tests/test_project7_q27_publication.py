from __future__ import annotations

import numpy as np
import pytest

from rtc.project7_q27_publication import (
    build_locked_event_provenance,
    bootstrap_mean_ci,
    paired_tfv_statistics,
    select_outcome_unexposed_events,
    validate_final_table_event_set,
)


def test_locked_event_provenance_binds_registry_inp_and_metadata(tmp_path) -> None:
    inp = tmp_path / "event_a.inp"
    inp.write_text("[TITLE]\nEvent A\n", encoding="utf-8")
    from rtc.project7_q27_publication import sha256_file

    row = {
        "event_id": "A",
        "inp_path": str(inp),
        "prepared_inp_sha256": sha256_file(inp),
        "rainfall_family": "chicago",
        "return_period_year": "10",
        "duration_minutes": "60",
    }
    provenance = build_locked_event_provenance([row], ["A"])
    assert provenance[0]["event_id"] == "A"
    assert provenance[0]["inp_sha256"] == row["prepared_inp_sha256"]
    assert provenance[0]["forcing_metadata"]["rainfall_family"] == "chicago"
    assert provenance[0]["duration_minutes"] == "60"


def test_locked_event_provenance_rejects_registry_hash_mismatch(tmp_path) -> None:
    inp = tmp_path / "event_a.inp"
    inp.write_text("[TITLE]\nEvent A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="INP SHA256"):
        build_locked_event_provenance(
            [{"event_id": "A", "inp_path": str(inp), "prepared_inp_sha256": "0" * 64}],
            ["A"],
        )


def test_prepared_and_step2_exposure_do_not_block_final() -> None:
    selected, report = select_outcome_unexposed_events(
        ["A", "B", "C"],
        [
            {"event_id": "A", "kind": "STEP2_TRAINING"},
            {"event_id": "B", "kind": "DEVELOPMENT_STEERING"},
        ],
    )
    assert selected == ["A", "C"]
    assert report["blocked_events"] == {"B": ["DEVELOPMENT_STEERING"]}


def test_paired_statistics_are_event_balanced_and_deterministic() -> None:
    proposed = [90.0, 180.0, 270.0, 360.0]
    baseline = [100.0, 200.0, 300.0, 400.0]
    first = paired_tfv_statistics(proposed, baseline, repetitions=1000, seed=42)
    second = paired_tfv_statistics(proposed, baseline, repetitions=1000, seed=42)
    assert first == second
    assert first["wins"] == 4
    assert first["event_balanced_mean_reduction_pct"] == pytest.approx(10.0)
    assert first["aggregate_volume_reduction_pct"] == pytest.approx(10.0)
    assert first["claim_classification"] == "SUPERIOR"


def test_bootstrap_ci_does_not_require_positive_result() -> None:
    low, high = bootstrap_mean_ci([-2.0, -1.0, 1.0, 2.0], repetitions=1000, seed=7)
    assert np.isfinite([low, high]).all()
    assert low <= high


def test_final_table_must_equal_locked_event_panel() -> None:
    validate_final_table_event_set([{"event_id": "A"}, {"event_id": "B"}], ["A", "B"])
    with pytest.raises(ValueError):
        validate_final_table_event_set([{"event_id": "A"}], ["A", "B"])
