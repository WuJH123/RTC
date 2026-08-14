from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from rtc.baselines import (
    FORMAL_FIXED_BASELINE_IDS,
    baseline_sensor_nodes,
    canonical_baseline_id,
    fixed_baseline_controller,
)
from rtc.closed_loop import CausalObservation
from rtc.formal_final_v5 import EXPECTED_STRATEGIES
from rtc.rule_baselines import AUTO_RBC_CONTRACT, EFD_CONTRACT


INP = """
[TITLE]
rule baseline test
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
J1 0 2 0 0 0
J2 0 2 0 0 0
[STORAGE]
S1 0 2 0 FUNCTIONAL 10 0 0 0 0
S2 0 2 0 FUNCTIONAL 10 0 0 0 0
[PUMPS]
P1 S1 J1 C1 ON
P2 S2 J2 C1 ON
[ORIFICES]
O1 J1 S1 SIDE 0.0 0.6 NO 0
"""


def _inp(tmp_path: Path) -> Path:
    path = tmp_path / "network.inp"
    path.write_text(INP, encoding="utf-8")
    return path


def _obs(sensor_ids: tuple[str, ...], sensor_depths: list[float]) -> CausalObservation:
    return CausalObservation(
        elapsed_seconds=600,
        current_time=datetime(2026, 1, 1, 0, 10),
        sensor_ids=sensor_ids,
        sensor_depth_m=np.asarray(sensor_depths, dtype=float),
        sensor_head_m=np.asarray(sensor_depths, dtype=float),
        actuator_ids=("P1", "P2", "O1"),
        actuator_target_setting=np.asarray([0.5, 0.5, 0.5]),
        actuator_current_setting=np.asarray([0.5, 0.5, 0.5]),
        actuator_flow_m3s=np.zeros(3),
        rainfall_node_ids=("J1", "J2", "S1", "S2"),
        observed_rainfall_mmhr=np.zeros(4),
    )


def test_rule_baselines_are_formal_and_alias_auto_rrc() -> None:
    assert canonical_baseline_id("Auto-RRC") == "auto_rbc"
    assert canonical_baseline_id("Auto-RBC") == "auto_rbc"
    assert "auto_rbc" in FORMAL_FIXED_BASELINE_IDS
    assert "efd" in FORMAL_FIXED_BASELINE_IDS
    assert EXPECTED_STRATEGIES == (
        "proposed",
        "no_control",
        "internal_rtc",
        "auto_rbc",
        "efd",
        "all_open",
        "all_closed",
    )


def test_auto_rbc_opens_high_upstream_storage_more(tmp_path: Path) -> None:
    path = _inp(tmp_path)
    sensors = baseline_sensor_nodes("auto_rbc", path)
    depth_map = {"S1": 1.8, "S2": 0.4, "J1": 0.2, "J2": 0.2}
    obs = _obs(sensors, [depth_map[n] for n in sensors])
    controller = fixed_baseline_controller("auto_rbc", inp_path=path)
    action = controller(obs)
    assert action.source == "AUTO_RBC_V1"
    assert action.diagnostics is not None
    assert action.diagnostics["rule_contract"] == AUTO_RBC_CONTRACT
    assert action.settings["P1"] > action.settings["P2"]
    assert all(0.0 <= value <= 1.0 for value in action.settings.values())


def test_efd_gives_more_discharge_to_more_volume_filled_storage(tmp_path: Path) -> None:
    path = _inp(tmp_path)
    sensors = baseline_sensor_nodes("efd", path)
    assert sensors == ("S1", "S2")
    obs = _obs(sensors, [1.8, 0.6])
    controller = fixed_baseline_controller("efd", inp_path=path)
    action = controller(obs)
    assert action.source == "EFD_V2"
    assert action.diagnostics is not None
    assert action.diagnostics["rule_contract"] == EFD_CONTRACT
    assert action.settings["P1"] > action.settings["P2"]
    assert action.settings["O1"] == 0.5
    assert float(action.diagnostics["filling_degree_std"]) > 0.0
    assert int(action.diagnostics["volume_based_storage_count"]) == 2
    assert int(action.diagnostics["depth_fallback_storage_count"]) == 0


def test_rule_baseline_move_limit_matches_controller_limit(tmp_path: Path) -> None:
    path = _inp(tmp_path)
    sensors = baseline_sensor_nodes("efd", path)
    obs = _obs(sensors, [2.0, 0.0])
    controller = fixed_baseline_controller(
        "efd", inp_path=path, max_delta_per_update=0.1
    )
    action = controller(obs)
    assert max(abs(float(value) - 0.5) for value in action.settings.values()) <= 0.1000001
