from __future__ import annotations

from pathlib import Path

import pytest

from rtc.inp_runtime import assert_native_controls_disabled, build_runtime_inp, section_has_payload


def test_python_runtime_removes_controls_and_rules(tmp_path: Path) -> None:
    source = tmp_path / "event.inp"
    runtime = tmp_path / "runtime.inp"
    source.write_text(
        "[OPTIONS]\nFLOW_UNITS CMS\nTHREADS 2\n"
        "[CONTROLS]\nPUMP P1 STATUS = ON\n"
        "[RULES]\nRULE R1\nIF NODE J1 DEPTH > 0.5\n",
        encoding="utf-8",
    )
    build_runtime_inp(source, runtime, native_controls=False, swmm_threads=1)
    assert not section_has_payload(runtime, "CONTROLS")
    assert not section_has_payload(runtime, "RULES")
    text = runtime.read_text(encoding="utf-8")
    assert "PUMP P1" not in text
    assert "RULE R1" not in text


def test_disabled_assertion_rejects_rules_only(tmp_path: Path) -> None:
    path = tmp_path / "rules.inp"
    path.write_text("[RULES]\nRULE R1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CONTROLS.*RULES"):
        assert_native_controls_disabled(path)
