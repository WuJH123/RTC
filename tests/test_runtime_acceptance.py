from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rtc.code_contract import rtc_source_tree_sha256
from rtc.project7_contract import PRODUCTION_CONTROLLER_CONTRACT
from rtc.runtime_acceptance import build_runtime_acceptance
from rtc.runtime_controller_guard import CONTINUITY_GUARD_CONTRACT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_acceptance_requires_exact_control_grid_budget_and_continuity(tmp_path: Path) -> None:
    config = tmp_path / "controller.json"
    config.write_text(
        json.dumps(
            {
                "contract": PRODUCTION_CONTROLLER_CONTRACT,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "record_stride_seconds": 300,
                "control_start_minutes": 60,
                "methodology_testbed": {
                    "claim_scope": "IDEALIZED_METHODOLOGY_TESTBED_NOT_FIELD_DIGITAL_TWIN",
                    "effective_warmup_minutes": 120,
                    "dwf_background_loading": True,
                    "baseline_true_state_advantage": ["internal_rtc", "auto_rbc", "efd"],
                },
                "controller": {
                    "history_steps": 13,
                    "horizon_steps": 72,
                    "decision_runtime_budget_seconds": 120.0,
                    "max_setting_delta_per_update": 0.5,
                    "enforce_cross_decision_target_continuity": True,
                    "enforce_sequential_horizon_continuity": True,
                },
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "run.decisions.jsonl"
    decisions.write_text(
        "".join(
            json.dumps(
                {
                    "elapsed_seconds": elapsed,
                    "source": "MPC",
                    "settings": {"A": 0.5},
                    "diagnostics": {
                        "decision_runtime_seconds": runtime,
                        "continuity_guard_contract": CONTINUITY_GUARD_CONTRACT,
                        "continuity_guard_passed": True,
                        "command_delta_from_current_max": 0.25,
                        "command_delta_from_previous_target_max": 0.25,
                        "planned_sequence_max_step_delta": 0.5,
                    },
                }
            )
            + "\n"
            for elapsed, runtime in [(3600, 8.0), (4200, 9.0), (4800, 7.5)]
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "run.json"
    metadata.write_text(
        json.dumps(
            {
                "data_contract": "CLOSED_LOOP_COMPACT_V2",
                "rtc_source_tree_sha256": rtc_source_tree_sha256(),
                "strategy": "proposed",
                "controller_config_sha256": _sha(config),
                "step1_model_sha256": "a" * 64,
                "step2_model_sha256": "b" * 64,
                "initial_observation_elapsed_seconds": 0,
                "control_start_minutes": 60,
                "control_update_seconds": 600,
                "observation_update_seconds": 300,
                "record_stride_seconds": 300,
                "prepared_event_clock": {"effective_warmup_minutes": 120.0},
                "project7_runtime_contract": {"contract": "fixture"},
                "decision_file": decisions.name,
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "strategy": "proposed",
                "metadata_path": str(metadata),
            }
        ]
    ).to_csv(index, index=False)

    evidence = build_runtime_acceptance(
        run_index_path=index,
        controller_config_path=config,
    )
    assert evidence["passed"] is True
    assert evidence["rtc_source_tree_sha256"] == rtc_source_tree_sha256()
    assert evidence["metrics"]["control_grid_violations"] == 0
    assert evidence["metrics"]["cross_decision_continuity_violations"] == 0
    assert evidence["metrics"]["planned_horizon_continuity_violations"] == 0
    assert evidence["metrics"]["fatal_runtime_fallbacks"] == 0
    assert evidence["metrics"]["decision_runtime_max_seconds"] == 9.0
