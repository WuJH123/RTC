from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtc.policy_artifacts_cli import _CANONICAL_RELATIVE, build_policy_artifact_map


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_policy_artifact_builder_writes_complete_utf8_json(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    for relative in _CANONICAL_RELATIVE.values():
        _touch(root / relative)
    inp = tmp_path / "wuhan.inp"
    priority = tmp_path / "priority.txt"
    sensors = tmp_path / "sensors.txt"
    plan = tmp_path / "baseline.json"
    for path in (inp, priority, sensors, plan):
        _touch(path)

    out = root / "policy_lock" / "artifacts.json"
    artefacts = build_policy_artifact_map(
        root=root,
        frozen_inp=inp,
        priority_nodes=priority,
        sensor_layout=sensors,
        baseline_plan=plan,
        output_path=out,
    )
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded == artefacts
    assert set(artefacts) == set(_CANONICAL_RELATIVE) | {
        "frozen_inp",
        "priority_nodes",
        "sensor_layout",
        "baseline_plan",
    }


def test_policy_artifact_builder_fails_on_missing_required_file(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    inp = tmp_path / "wuhan.inp"
    priority = tmp_path / "priority.txt"
    sensors = tmp_path / "sensors.txt"
    plan = tmp_path / "baseline.json"
    for path in (inp, priority, sensors, plan):
        _touch(path)
    with pytest.raises(ValueError, match="required files are missing"):
        build_policy_artifact_map(
            root=root,
            frozen_inp=inp,
            priority_nodes=priority,
            sensor_layout=sensors,
            baseline_plan=plan,
            output_path=root / "policy_lock" / "artifacts.json",
        )
