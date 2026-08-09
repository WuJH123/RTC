from __future__ import annotations

import pandas as pd

from rtc.data_index import build_d2_run_index


def test_d2_center_branch_is_not_duplicated_per_probe_actuator() -> None:
    manifest = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "e1:t3600",
                "checkpoint_minutes": 60,
                "actuator_id": "P1",
                "candidate_action_sha256": "base",
            },
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "e1:t3600",
                "checkpoint_minutes": 60,
                "actuator_id": "P2",
                "candidate_action_sha256": "base",
            },
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "e1:t3600",
                "checkpoint_minutes": 60,
                "actuator_id": "P1",
                "candidate_action_sha256": "p1_up",
            },
        ]
    )
    runs = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "checkpoint_id": "e1:t3600",
                "checkpoint_minutes": 60,
                "candidate_action_sha256": "base",
                "metadata_path": "base.json",
            },
            {
                "event_id": "e1",
                "rainfall_group": "g1",
                "checkpoint_id": "e1:t3600",
                "checkpoint_minutes": 60,
                "candidate_action_sha256": "p1_up",
                "metadata_path": "p1_up.json",
            },
        ]
    )
    result = build_d2_run_index(manifest, runs)
    assert len(result) == 2
    base = result[result["candidate_action_sha256"] == "base"].iloc[0]
    assert int(base["manifest_rows_collapsed"]) == 2
    assert base["probe_actuator_ids_json"] == '["P1","P2"]'
    assert base["scientific_split"] == "development"
    assert base["development_fold"] == "train"
