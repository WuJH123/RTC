from __future__ import annotations

from pathlib import Path

from rtc.inp_lineage import physical_contract_sha256


BASE = """
[OPTIONS]
FLOW_UNITS CMS
START_DATE 01/01/2020
END_DATE 01/02/2020
[JUNCTIONS]
J1 10 2 0 0 0
J2 9 2 0 0 0
[CONDUITS]
C1 J1 J2 100 0.01 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 1.0 0 0 0 1
[PUMPS]
P1 J2 J1 CURVE1 ON
[CURVES]
CURVE1 Pump3 0 0
CURVE1 1 1
[RAINGAGES]
RG1 INTENSITY 0:05 1.0 TIMESERIES TS1
[TIMESERIES]
TS1 01/01/2020 00:00 0
TS1 01/01/2020 00:05 10
[CONTROLS]
RULE R1
IF NODE J1 DEPTH > 1
THEN PUMP P1 SETTING = 0.8
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_physical_hash_ignores_event_forcing_and_controls(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.inp", BASE)
    b_text = BASE.replace("00:05 10", "00:05 50").replace(
        "THEN PUMP P1 SETTING = 0.8", "THEN PUMP P1 SETTING = 0.2"
    ).replace("END_DATE 01/02/2020", "END_DATE 01/03/2020")
    b = _write(tmp_path / "b.inp", b_text)
    assert physical_contract_sha256(a) == physical_contract_sha256(b)


def test_physical_hash_changes_when_hydraulic_asset_changes(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.inp", BASE)
    b = _write(tmp_path / "b.inp", BASE.replace("C1 CIRCULAR 1.0", "C1 CIRCULAR 1.2"))
    assert physical_contract_sha256(a) != physical_contract_sha256(b)
