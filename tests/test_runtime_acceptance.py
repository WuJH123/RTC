from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rtc.code_contract import rtc_source_tree_sha256
from rtc.runtime_acceptance import build_runtime_acceptance


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_acceptance_requires_exact_control_grid_and_budget(tmp_path: Path) -> None:
    config = tmp_path / "controller.json"
    config.write_text(
        json.dumps(
            {
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "record_stride_seconds": 300,
                "control_start_minutes": 60,
                "controller": {
                    "history_steps": 13,
                    "horizon_steps": 24,
                    "decision_runtime_budget_seconds": 120.0,
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
                    "diagnostics": {"decision_runtime_seconds": runtime},
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
    assert evidence["metrics"]["fatal_runtime_fallbacks"] == 0
    assert evidence["metrics"]["decision_runtime_max_seconds"] == 9.0
