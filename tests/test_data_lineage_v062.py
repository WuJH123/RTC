from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtc.data_design import design_multi_actuator_rollouts
from rtc.inp import Actuator, ActuatorCatalog
from rtc.inp_lineage import scientific_event_contract_sha256
from rtc.inp_runtime import build_runtime_inp
from rtc.replay_prefix import load_checkpoint_reference, verify_replayed_checkpoint


def _event_inp(path: Path, *, rain_value: float, controls: str = "", threads: int = 1) -> None:
    path.write_text(
        "\n".join(
            [
                "[OPTIONS]",
                "FLOW_UNITS CMS",
                f"THREADS {threads}",
                "START_DATE 01/01/2020",
                "END_DATE 01/02/2020",
                "[JUNCTIONS]",
                "N1 0 2 0 0 0",
                "N2 0 2 0 0 0",
                "[CONDUITS]",
                "C1 N1 N2 10 0.01 0 0 0 0",
                "[TIMESERIES]",
                f"R1 01/01/2020 00:00 {rain_value}",
                "[CONTROLS]",
                controls,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_event_contract_ignores_controls_and_threads_but_not_rainfall(tmp_path: Path) -> None:
    a = tmp_path / "a.inp"
    b = tmp_path / "b.inp"
    c = tmp_path / "c.inp"
    _event_inp(a, rain_value=10.0, controls="RULE R1", threads=1)
    _event_inp(b, rain_value=10.0, controls="RULE DIFFERENT", threads=8)
    _event_inp(c, rain_value=11.0, controls="RULE R1", threads=1)
    assert scientific_event_contract_sha256(a) == scientific_event_contract_sha256(b)
    assert scientific_event_contract_sha256(a) != scientific_event_contract_sha256(c)


def test_event_contract_hashes_external_file_bytes(tmp_path: Path) -> None:
    rain = tmp_path / "rain.dat"
    rain.write_text("2020-01-01 00:00 1\n", encoding="utf-8")
    inp = tmp_path / "event.inp"
    inp.write_text(
        "[OPTIONS]\nFLOW_UNITS CMS\nTHREADS 1\n[TIMESERIES]\nR1 FILE rain.dat\n",
        encoding="utf-8",
    )
    first = scientific_event_contract_sha256(inp)
    rain.write_text("2020-01-01 00:00 2\n", encoding="utf-8")
    assert scientific_event_contract_sha256(inp) != first


def test_runtime_relocation_preserves_relative_external_forcing(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rain = source_dir / "rain series.dat"
    rain.write_text("2020-01-01 00:00 1\n", encoding="utf-8")
    inp = source_dir / "event.inp"
    inp.write_text(
        "[OPTIONS]\nFLOW_UNITS CMS\nTHREADS 2\n"
        "[TIMESERIES]\nR1 FILE \"rain series.dat\"\n"
        "[CONTROLS]\nRULE X\n",
        encoding="utf-8",
    )
    expected = scientific_event_contract_sha256(inp)
    runtime = tmp_path / "elsewhere" / "runtime.inp"
    build_runtime_inp(inp, runtime, native_controls=False, swmm_threads=1)
    text = runtime.read_text(encoding="utf-8")
    assert str(rain.resolve()) in text
    assert scientific_event_contract_sha256(runtime) == expected


def test_replay_prefix_verifies_complete_state_and_readback(tmp_path: Path) -> None:
    compact = tmp_path / "base.npz"
    state = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
    settings = np.asarray([[0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    np.savez_compressed(
        compact,
        elapsed_seconds=np.asarray([0, 300], dtype=np.int64),
        node_ids=np.asarray(["N1", "N2"]),
        actuator_ids=np.asarray(["A1", "A2"]),
        state_si=state,
        current_setting=settings,
    )
    meta = tmp_path / "base.json"
    meta.write_text(
        json.dumps({"compact_file": compact.name, "swmm_engine_version": "5.2.4"}),
        encoding="utf-8",
    )
    reference = load_checkpoint_reference(meta, elapsed_seconds=300)
    evidence = verify_replayed_checkpoint(
        reference,
        elapsed_seconds=300,
        node_ids=("N1", "N2"),
        actuator_ids=("A1", "A2"),
        state_si=state[1],
        current_setting=settings[1],
        swmm_engine_version="5.2.4",
    )
    assert evidence["passed"] is True
    changed = state[1].copy()
    changed[0, 0] += 1e-3
    with pytest.raises(ValueError, match="replay state diverged"):
        verify_replayed_checkpoint(
            reference,
            elapsed_seconds=300,
            node_ids=("N1", "N2"),
            actuator_ids=("A1", "A2"),
            state_si=changed,
            current_setting=settings[1],
            swmm_engine_version="5.2.4",
        )


def test_d3_sequences_respect_frozen_setting_delta() -> None:
    catalog = ActuatorCatalog(
        (
            Actuator("A1", "pump", "N1", "N2"),
            Actuator("A2", "orifice", "N2", "N3"),
        )
    )
    checkpoints = pd.DataFrame(
        [{"checkpoint_id": "e:t600", "setting:A1": 0.5, "setting:A2": 0.5}]
    )
    manifest = design_multi_actuator_rollouts(
        checkpoints,
        catalog,
        horizon_steps=8,
        sequences_per_checkpoint=4,
        perturbation_std=1.0,
        change_probability=1.0,
        max_delta_per_update=0.1,
        seed=1,
        include_hold=False,
    )
    for raw in manifest["settings_sequence_json"]:
        sequence = json.loads(raw)
        previous = np.asarray([0.5, 0.5], dtype=float)
        for step in sequence:
            current = np.asarray([step["A1"], step["A2"]], dtype=float)
            assert np.max(np.abs(current - previous)) <= 0.1000001
            previous = current
