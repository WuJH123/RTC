from __future__ import annotations

import json
from pathlib import Path

from rtc.baseline_cache import baseline_cache_key, validate_fixed_baseline_run
from rtc.inp_lineage import physical_contract_sha256


def _inp(tmp_path: Path, *, controls: bool) -> Path:
    path = tmp_path / ("internal.inp" if controls else "no_control.inp")
    text = """
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
N1 0 1 0 0 0
N2 0 1 0 0 0
[CONDUITS]
C1 N1 N2 10 0.01 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1
[CONTROLS]
"""
    if controls:
        text += "RULE R1\nIF NODE N1 DEPTH > 0.5\nTHEN CONDUIT C1 STATUS = CLOSED\n"
    path.write_text(text, encoding="utf-8")
    return path


def _evidence(tmp_path: Path, inp: Path, *, controller_present: bool, decisions: list[dict]) -> Path:
    compact = tmp_path / "run.compact.npz"
    compact.write_bytes(b"compact")
    stats = tmp_path / "run.node_statistics.csv.gz"
    stats.write_bytes(b"stats")
    decision = tmp_path / "run.decisions.jsonl"
    decision.write_text("".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8")
    meta = tmp_path / "run.json"
    meta.write_text(
        json.dumps(
            {
                "data_contract": "CLOSED_LOOP_COMPACT_V2",
                "inp_path": str(inp),
                "controller_present": controller_present,
                "compact_file": compact.name,
                "node_statistics_file": stats.name,
                "decision_file": decision.name,
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_baseline_cache_key_invalidates_when_runtime_contract_changes() -> None:
    base = dict(
        source_inp_sha256="a" * 64,
        physical_network_sha256="b" * 64,
        strategy="no_control",
        model_step_seconds=300,
        control_update_seconds=600,
        record_stride_seconds=300,
        control_start_minutes=60,
        swmm_threads_per_process=1,
    )
    key1 = baseline_cache_key(**base)
    key2 = baseline_cache_key(**{**base, "record_stride_seconds": 600})
    key3 = baseline_cache_key(**{**base, "strategy": "internal_rtc"})
    assert key1 != key2
    assert key1 != key3


def test_no_control_and_internal_rtc_semantics_are_fail_closed(tmp_path: Path) -> None:
    no_control = _inp(tmp_path, controls=False)
    no_meta = _evidence(tmp_path, no_control, controller_present=False, decisions=[])
    result = validate_fixed_baseline_run(
        strategy="no_control",
        main_metadata_path=no_meta,
        source_physical_sha256=physical_contract_sha256(no_control),
    )
    assert result["native_controls_enabled"] is False
    assert result["decision_count"] == 0

    internal_dir = tmp_path / "internal_case"
    internal_dir.mkdir()
    internal = _inp(internal_dir, controls=True)
    internal_meta = _evidence(internal_dir, internal, controller_present=False, decisions=[])
    result = validate_fixed_baseline_run(
        strategy="internal_rtc",
        main_metadata_path=internal_meta,
        source_physical_sha256=physical_contract_sha256(internal),
    )
    assert result["native_controls_enabled"] is True
    assert result["decision_count"] == 0


def test_static_baselines_require_exact_logged_settings(tmp_path: Path) -> None:
    runtime = _inp(tmp_path, controls=False)
    physical = physical_contract_sha256(runtime)

    all_open_dir = tmp_path / "open"
    all_open_dir.mkdir()
    open_inp = _inp(all_open_dir, controls=False)
    open_meta = _evidence(
        all_open_dir,
        open_inp,
        controller_present=True,
        decisions=[
            {"source": "ALL_OPEN", "settings": {"A": 1.0, "B": 1.0}},
            {"source": "ALL_OPEN", "settings": {"A": 1.0, "B": 1.0}},
        ],
    )
    validate_fixed_baseline_run(
        strategy="all_open",
        main_metadata_path=open_meta,
        source_physical_sha256=physical_contract_sha256(open_inp),
    )

    hold_dir = tmp_path / "hold"
    hold_dir.mkdir()
    hold_inp = _inp(hold_dir, controls=False)
    hold_meta = _evidence(
        hold_dir,
        hold_inp,
        controller_present=True,
        decisions=[
            {"source": "FROZEN_HOLD", "settings": {"A": 0.2, "B": 0.8}},
            {"source": "FROZEN_HOLD", "settings": {"A": 0.2, "B": 0.8}},
        ],
    )
    validate_fixed_baseline_run(
        strategy="hold",
        main_metadata_path=hold_meta,
        source_physical_sha256=physical_contract_sha256(hold_inp),
    )

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_inp = _inp(bad_dir, controls=False)
    bad_meta = _evidence(
        bad_dir,
        bad_inp,
        controller_present=True,
        decisions=[{"source": "ALL_OPEN", "settings": {"A": 1.0, "B": 0.9}}],
    )
    try:
        validate_fixed_baseline_run(
            strategy="all_open",
            main_metadata_path=bad_meta,
            source_physical_sha256=physical_contract_sha256(bad_inp),
        )
    except ValueError as exc:
        assert "did not command every eligible actuator" in str(exc)
    else:
        raise AssertionError("invalid All-open evidence was accepted")
