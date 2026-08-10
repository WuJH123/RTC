from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rtc.event_preparation import prepare_event_inp
from rtc.phase0_timescale import _branch_series
from rtc.simulation_assets import (
    SimulationAssetRegistry,
    assert_endpoint_available,
    d2_identity,
    event_prefix_family_sha256,
    sha256_json,
)


def _inp(path: Path, *, start_time: str = "00:00:00", end_time: str = "12:00:00") -> Path:
    text = f"""[OPTIONS]
FLOW_UNITS           CMS
START_DATE           01/01/2020
START_TIME           {start_time}
REPORT_START_DATE    01/01/2020
REPORT_START_TIME    {start_time}
END_DATE             01/01/2020
END_TIME             {end_time}
REPORT_END_DATE      01/01/2020
REPORT_END_TIME      {end_time}

[JUNCTIONS]
N1 0 5 0 0 0

[RAINGAGES]
RG1 INTENSITY 0:05 1.0 TIMESERIES TS1

[TIMESERIES]
TS1 01/01/2020 00:55:00 0
TS1 01/01/2020 01:00:00 1
TS1 01/01/2020 01:05:00 0

[END]
"""
    path.write_text(text, encoding="utf-8")
    return path


def _reference(tmp_path: Path) -> Path:
    compact = tmp_path / "reference.compact.npz"
    np.savez_compressed(
        compact,
        elapsed_seconds=np.asarray([0, 3600], dtype=np.int64),
        node_ids=np.asarray(["N1"]),
        state_si=np.asarray(
            [[[0, 0, 0, 0, 0, 0]], [[1, 1, 0, 1, 1, 1]]], dtype=np.float32
        ),
        actuator_ids=np.asarray(["P1"]),
        current_setting=np.asarray([[0.0], [0.5]], dtype=np.float32),
    )
    metadata = tmp_path / "reference.json"
    metadata.write_text(
        json.dumps(
            {
                "compact_file": compact.name,
                "swmm_engine_version": "5.2.4",
            }
        ),
        encoding="utf-8",
    )
    return metadata


def test_target_effective_warmup_accounts_for_existing_source_prefix(tmp_path: Path) -> None:
    source = _inp(tmp_path / "source.inp")
    prepared = tmp_path / "prepared.inp"
    evidence = prepare_event_inp(
        source,
        prepared,
        target_effective_warmup_minutes=120,
        post_rain_tail_minutes=360,
    )
    assert evidence["source_pre_rain_prefix_minutes"] == pytest.approx(60.0)
    assert evidence["additional_warmup_minutes"] == 60
    assert evidence["effective_warmup_minutes"] == pytest.approx(120.0)
    assert evidence["rainfall_onset_elapsed_minutes"] == pytest.approx(120.0)


def test_event_family_ignores_only_recovery_tail(tmp_path: Path) -> None:
    a = _inp(tmp_path / "a.inp", end_time="10:00:00")
    b = _inp(tmp_path / "b.inp", end_time="12:00:00")
    c = _inp(tmp_path / "c.inp", start_time="23:00:00", end_time="12:00:00")
    text = c.read_text(encoding="utf-8").replace(
        "START_DATE           01/01/2020", "START_DATE           12/31/2019"
    ).replace(
        "REPORT_START_DATE    01/01/2020", "REPORT_START_DATE    12/31/2019"
    )
    c.write_text(text, encoding="utf-8")
    assert event_prefix_family_sha256(a) == event_prefix_family_sha256(b)
    assert event_prefix_family_sha256(a) != event_prefix_family_sha256(c)


def test_endpoint_preflight_fails_before_swmm(tmp_path: Path) -> None:
    short = _inp(tmp_path / "short.inp", end_time="02:00:00")
    with pytest.raises(ValueError, match="endpoint preflight failed before SWMM launch"):
        assert_endpoint_available(short, checkpoint_seconds=90 * 60, horizon_seconds=60 * 60)


def test_d2_identity_is_tail_independent_but_horizon_specific(tmp_path: Path) -> None:
    a = _inp(tmp_path / "a.inp", end_time="10:00:00")
    b = _inp(tmp_path / "b.inp", end_time="12:00:00")
    reference = _reference(tmp_path)
    key_a, family_a, _ = d2_identity(
        inp_path=a,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=60,
        horizon_seconds=210 * 60,
    )
    key_b, family_b, _ = d2_identity(
        inp_path=b,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=60,
        horizon_seconds=210 * 60,
    )
    key_long, family_long, _ = d2_identity(
        inp_path=b,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=60,
        horizon_seconds=360 * 60,
    )
    assert family_a == family_b == family_long
    assert key_a == key_b
    assert key_a != key_long


def test_local_registry_verifies_referenced_artifacts(tmp_path: Path) -> None:
    compact = tmp_path / "branch.compact.npz"
    compact.write_bytes(b"compact")
    stats = tmp_path / "branch.stats.csv.gz"
    stats.write_bytes(b"stats")
    metadata = tmp_path / "branch.json"
    metadata.write_text(
        json.dumps({"compact_file": compact.name, "node_statistics_file": stats.name}),
        encoding="utf-8",
    )
    registry = SimulationAssetRegistry(tmp_path / "assets")
    identity = {"x": 1}
    key = sha256_json({"identity": identity, "horizon": 60})
    family = sha256_json(identity)
    registry.register(
        simulation_key=key,
        family_key=family,
        kind="TEST",
        horizon_seconds=60,
        metadata_path=metadata,
        identity=identity,
    )
    assert registry.lookup_exact(key) is not None
    compact.write_bytes(b"changed")
    assert registry.lookup_exact(key) is None


def test_long_d2_trajectory_can_supply_shorter_timing_view(tmp_path: Path) -> None:
    checkpoint = 60 * 60
    source_horizon = 360 * 60
    step = 60
    times = np.arange(checkpoint, checkpoint + source_horizon + step, step, dtype=np.int64)
    n = len(times)
    compact = tmp_path / "long.compact.npz"
    state = np.zeros((n, 1, 6), dtype=np.float32)
    state[:, 0, 0] = np.linspace(0, 1, n)
    np.savez_compressed(
        compact,
        elapsed_seconds=times,
        actuator_ids=np.asarray(["P1"]),
        actuator_flow_m3s=np.zeros((n, 1), dtype=np.float32),
        current_setting=np.zeros((n, 1), dtype=np.float32),
        state_si=state,
    )
    metadata = tmp_path / "long.json"
    metadata.write_text(
        json.dumps(
            {
                "compact_file": compact.name,
                "checkpoint_minutes": 60,
                "horizon_minutes": 360,
            }
        ),
        encoding="utf-8",
    )
    t, *_rest, source_horizon_seconds = _branch_series(
        metadata, "P1", analysis_horizon_minutes=210
    )
    assert int(t[-1] - t[0]) == 210 * 60
    assert source_horizon_seconds == 360 * 60
