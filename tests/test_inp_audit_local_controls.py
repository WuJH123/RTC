from __future__ import annotations

from pathlib import Path

from rtc.inp_audit_v2 import audit_inp


def test_no_supervisory_rtc_contract_reports_intrinsic_pump_startup_shutoff(tmp_path: Path) -> None:
    inp = tmp_path / "model.inp"
    inp.write_text(
        """
[OPTIONS]
FLOW_UNITS CMS
FLOW_ROUTING DYNWAVE
[JUNCTIONS]
N1 0 10 0 0 0
N2 0 10 0 0 0
[PUMPS]
P1 N1 N2 C1 OFF 5 4
[CURVES]
C1 PUMP2 0 0
C1 1 1
[CONTROLS]
RULE R1
IF NODE N1 DEPTH > 6
THEN PUMP P1 STATUS = ON
""",
        encoding="utf-8",
    )
    result = audit_inp(inp, ("N1",))
    assert result["contract"] == "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC"
    assert result["native_controlled_actuators"] == 1
    assert result["pump_intrinsic_startup_shutoff_count"] == 1
    local = result["pump_intrinsic_startup_shutoff_controls"][0]
    assert local["pump_id"] == "P1"
    assert local["startup_depth_native"] == 5.0
    assert local["shutoff_depth_native"] == 4.0
    assert result["no_control_contract"]["id"] == "NO_SUPERVISORY_RTC_V2"
