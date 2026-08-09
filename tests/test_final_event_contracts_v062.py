from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rtc.final_event_contracts import build_final_event_contracts, verify_final_event_contracts
from rtc.inp_lineage import physical_contract_sha256


def _event(path: Path, rain: float) -> None:
    path.write_text(
        "\n".join(
            [
                "[OPTIONS]",
                "FLOW_UNITS CMS",
                "THREADS 1",
                "[JUNCTIONS]",
                "N1 0 2 0 0 0",
                "N2 0 2 0 0 0",
                "[CONDUITS]",
                "C1 N1 N2 10 0.01 0 0 0 0",
                "[TIMESERIES]",
                f"R1 01/01/2020 00:00 {rain}",
                "[CONTROLS]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_final_event_bytes_are_frozen_before_policy_lock(tmp_path: Path) -> None:
    inp = tmp_path / "final.inp"
    _event(inp, 10.0)
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "event_id": "F1",
                "rainfall_group": "FG1",
                "inp_path": str(inp),
                "scientific_split": "final",
                "development_fold": "",
            }
        ]
    ).to_csv(registry, index=False)
    physical = physical_contract_sha256(inp)
    lock = tmp_path / "final_event_contracts.json"
    build_final_event_contracts(
        split_registry_path=registry,
        output_path=lock,
        expected_physical_sha256=physical,
    )
    assert verify_final_event_contracts(
        lock,
        split_registry_path=registry,
        expected_physical_sha256=physical,
        verify_current_files=True,
    )["event_count"] == 1

    _event(inp, 11.0)
    with pytest.raises(ValueError, match="source INP bytes changed after lock"):
        verify_final_event_contracts(
            lock,
            split_registry_path=registry,
            expected_physical_sha256=physical,
            verify_current_files=True,
        )


def test_final_event_lock_rejects_registry_event_set_change(tmp_path: Path) -> None:
    inp = tmp_path / "final.inp"
    _event(inp, 10.0)
    registry = tmp_path / "registry.csv"
    base = pd.DataFrame(
        [
            {
                "event_id": "F1",
                "rainfall_group": "FG1",
                "inp_path": str(inp),
                "scientific_split": "final",
                "development_fold": "",
            }
        ]
    )
    base.to_csv(registry, index=False)
    physical = physical_contract_sha256(inp)
    lock = tmp_path / "final_event_contracts.json"
    build_final_event_contracts(
        split_registry_path=registry,
        output_path=lock,
        expected_physical_sha256=physical,
    )

    changed = pd.concat(
        [
            base,
            pd.DataFrame(
                [
                    {
                        "event_id": "F2",
                        "rainfall_group": "FG2",
                        "inp_path": str(inp),
                        "scientific_split": "final",
                        "development_fold": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    changed.to_csv(registry, index=False)
    with pytest.raises(ValueError, match="different split registry"):
        verify_final_event_contracts(
            lock,
            split_registry_path=registry,
            expected_physical_sha256=physical,
            verify_current_files=False,
        )
