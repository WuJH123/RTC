from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rtc.baseline_cache_cli import locked_final_contract, validate_final_event_registry
from rtc.code_contract import rtc_source_tree_sha256
from rtc.inp_lineage import physical_contract_sha256
from rtc.inp_runtime import sha256_file


def _inp(path: Path, diameter: float = 1.0) -> Path:
    path.write_text(
        f"""
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
N1 0 1 0 0 0
N2 0 1 0 0 0
[CONDUITS]
C1 N1 N2 10 0.01 0 0 0 0
[XSECTIONS]
C1 CIRCULAR {diameter} 0 0 0 1
[CONTROLS]
""",
        encoding="utf-8",
    )
    return path


def _lock(tmp_path: Path, source: Path) -> Path:
    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps(
            {
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "record_stride_seconds": 300,
                "control_start_minutes": 60,
                "exact_global_peak": False,
                "controller": {"history_steps": 13, "horizon_steps": 24},
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "baseline_plan.json"
    plan.write_text(
        json.dumps(
            {
                "contract": "FORMAL_BASELINE_PLAN_V5_DYNAMIC_RULE_COMPARATORS",
                "strategies": [
                    "proposed",
                    "no_control",
                    "internal_rtc",
                    "auto_rbc",
                    "efd",
                    "all_open",
                    "all_closed",
                ],
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "split.csv"
    rows = [{"rainfall_group": "g_dev", "scientific_split": "development"}]
    rows.extend(
        {"rainfall_group": f"g_final_{i:02d}", "scientific_split": "final"}
        for i in range(24)
    )
    pd.DataFrame(rows).to_csv(split, index=False)
    artefacts = {
        "controller_config": str(controller),
        "baseline_plan": str(plan),
        "split_registry": str(split),
    }
    lock = tmp_path / "policy_lock.json"
    lock.write_text(
        json.dumps(
            {
                "contract": "WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND",
                "rtc_source_tree_sha256": rtc_source_tree_sha256(),
                "physical_network_sha256": physical_contract_sha256(source),
                "artefacts": artefacts,
                "sha256": {name: sha256_file(path) for name, path in artefacts.items()},
            }
        ),
        encoding="utf-8",
    )
    return lock


def test_final_baseline_registry_must_match_locked_split_and_physics(tmp_path: Path) -> None:
    source = _inp(tmp_path / "event.inp")
    lock = _lock(tmp_path, source)
    config, strategies, physical, groups = locked_final_contract(lock)
    assert config.name == "controller.json"
    assert strategies == (
        "no_control",
        "internal_rtc",
        "auto_rbc",
        "efd",
        "all_open",
        "all_closed",
    )
    assert len(groups) == 24

    registry_rows = [
        {
            "event_id": "e_dev",
            "rainfall_group": "g_dev",
            "inp_path": str(source),
            "scientific_split": "development",
        }
    ]
    registry_rows.extend(
        {
            "event_id": f"e_final_{i:02d}",
            "rainfall_group": f"g_final_{i:02d}",
            "inp_path": str(source),
            "scientific_split": "final",
        }
        for i in range(24)
    )
    registry = pd.DataFrame(registry_rows)
    checked = validate_final_event_registry(
        registry,
        locked_physical_sha256=physical,
        locked_final_groups=groups,
    )
    assert len(checked) == 25

    wrong_group = registry.copy()
    wrong_group.loc[wrong_group.index[-1], "rainfall_group"] = "g_other"
    with pytest.raises(ValueError, match="differ from Policy Lock"):
        validate_final_event_registry(
            wrong_group,
            locked_physical_sha256=physical,
            locked_final_groups=groups,
        )

    wrong_source = _inp(tmp_path / "wrong.inp", diameter=2.0)
    wrong_physics = registry.copy()
    wrong_physics.loc[wrong_physics.index[-1], "inp_path"] = str(wrong_source)
    with pytest.raises(ValueError, match="physical network differs"):
        validate_final_event_registry(
            wrong_physics,
            locked_physical_sha256=physical,
            locked_final_groups=groups,
        )


def test_locked_baseline_plan_hash_is_enforced(tmp_path: Path) -> None:
    source = _inp(tmp_path / "event.inp")
    lock = _lock(tmp_path, source)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    plan = Path(payload["artefacts"]["baseline_plan"])
    plan.write_text(
        json.dumps({"strategies": ["proposed", "no_control"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact changed: baseline_plan"):
        locked_final_contract(lock)
