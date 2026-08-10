from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.graph import GraphSchema
from rtc.lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from rtc.swmm_data import STATE_CHANNELS


def _graph() -> GraphSchema:
    return GraphSchema(
        node_ids=("N1", "N2"),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.asarray(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        static_node_feature_names=(
            "invert_elevation_m",
            "max_depth_m",
            "is_junction",
            "is_outfall",
            "is_storage",
            "is_divider",
        ),
        actuator_ids=("A1",),
        actuator_upstream=np.asarray([0], dtype=np.int64),
        actuator_downstream=np.asarray([1], dtype=np.int64),
        actuator_physics=np.zeros((1, 19), dtype=np.float32),
        actuator_physics_feature_names=tuple(f"f{i}" for i in range(19)),
        system_units="SI",
    )


def _trajectory(tmp_path: Path) -> pd.DataFrame:
    times = np.arange(20, dtype=np.int64) * 300
    state = np.zeros((20, 2, 6), dtype=np.float32)
    # dry 0..7, wet 8..13, flood/high 14..19
    state[8:14, :, 0] = 0.35
    state[14:, :, 0] = 0.85
    state[16:, 0, 2] = 0.01
    rain = np.zeros((20, 2), dtype=np.float32)
    rain[8:14, :] = 5.0
    compact = tmp_path / "traj.compact.npz"
    np.savez_compressed(
        compact,
        elapsed_seconds=times,
        node_ids=np.asarray(["N1", "N2"]),
        state_channels=np.asarray(STATE_CHANNELS),
        state_si=state,
        rainfall_mmhr=rain,
        actuator_ids=np.asarray(["A1"]),
        target_setting=np.zeros((20, 1), dtype=np.float32),
        current_setting=np.zeros((20, 1), dtype=np.float32),
        actuator_flow_m3s=np.zeros((20, 1), dtype=np.float32),
    )
    meta = tmp_path / "traj.json"
    meta.write_text(
        json.dumps(
            {
                "compact_file": compact.name,
                "swmm_engine_version": "5.2.4",
            }
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(
        [
            {
                "metadata_path": str(meta),
                "event_id": "E1",
                "rainfall_group": "G1",
                "scientific_split": "development",
                "development_fold": "train",
            }
        ]
    )


def test_step1_dataset_classifies_dry_wet_and_high(tmp_path: Path) -> None:
    dataset = CausalStep1TrajectoryDataset(
        _trajectory(tmp_path),
        graph=_graph(),
        sensor_nodes=("N1",),
        history_steps=3,
        model_step_seconds=300,
    )
    assert dataset.stratum_counts["dry"] > 0
    assert dataset.stratum_counts["wet"] > 0
    assert dataset.stratum_counts["flood_or_high_depth"] > 0
    assert sum(dataset.stratum_counts.values()) == len(dataset)


def test_training_sampler_uses_fixed_stratified_trajectory_budget(tmp_path: Path) -> None:
    dataset = CausalStep1TrajectoryDataset(
        _trajectory(tmp_path),
        graph=_graph(),
        sensor_nodes=("N1",),
        history_steps=3,
        model_step_seconds=300,
    )
    sampler = TrajectoryBatchSampler(
        dataset,
        batch_size=4,
        seed=42,
        shuffle=True,
        stratified=True,
        samples_per_trajectory=20,
    )
    selected = [idx for batch in sampler for idx in batch]
    assert len(selected) == 20
    strata = [dataset.samples[idx].stratum for idx in selected]
    assert strata.count("dry") == 5
    assert strata.count("wet") == 7
    assert strata.count("flood_or_high_depth") == 8


def test_validation_sampler_still_visits_every_window_once(tmp_path: Path) -> None:
    dataset = CausalStep1TrajectoryDataset(
        _trajectory(tmp_path),
        graph=_graph(),
        sensor_nodes=("N1",),
        history_steps=3,
        model_step_seconds=300,
    )
    sampler = TrajectoryBatchSampler(
        dataset,
        batch_size=4,
        seed=0,
        shuffle=False,
        stratified=False,
    )
    selected = [idx for batch in sampler for idx in batch]
    assert selected == list(range(len(dataset)))
