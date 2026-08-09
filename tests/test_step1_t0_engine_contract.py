from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtc.graph import GraphSchema
from rtc.lazy_step1 import CausalStep1TrajectoryDataset


def _graph() -> GraphSchema:
    return GraphSchema(
        node_ids=("N1", "N2"),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.zeros((2, 2), dtype=np.float32),
        static_node_feature_names=("f1", "f2"),
        actuator_ids=("A1",),
        actuator_upstream=np.asarray([0], dtype=np.int64),
        actuator_downstream=np.asarray([1], dtype=np.int64),
        actuator_physics=np.zeros((1, 6), dtype=np.float32),
        actuator_physics_feature_names=("p1", "p2", "p3", "p4", "p5", "p6"),
        system_units="SI",
    )


def _trajectory(root: Path, name: str, times: list[int], engine: str) -> Path:
    compact = root / f"{name}.npz"
    n = len(times)
    np.savez_compressed(
        compact,
        elapsed_seconds=np.asarray(times, dtype=np.int64),
        node_ids=np.asarray(["N1", "N2"]),
        actuator_ids=np.asarray(["A1"]),
        state_si=np.zeros((n, 2, 6), dtype=np.float32),
        rainfall_mmhr=np.zeros((n, 2, 1), dtype=np.float32),
        current_setting=np.full((n, 1), 0.5, dtype=np.float32),
        actuator_flow_m3s=np.zeros((n, 1), dtype=np.float32),
    )
    meta = root / f"{name}.json"
    meta.write_text(
        json.dumps({"compact_file": compact.name, "swmm_engine_version": engine}),
        encoding="utf-8",
    )
    return meta


def _index(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metadata_path": str(path),
                "event_id": f"e{i}",
                "rainfall_group": f"g{i}",
                "scientific_split": "development",
                "development_fold": "train",
            }
            for i, path in enumerate(paths)
        ]
    )


def test_step1_requires_t0_first_frame(tmp_path: Path) -> None:
    bad = _trajectory(tmp_path, "bad", [300, 600, 900], "5.2.4")
    with pytest.raises(ValueError, match="t=0 frame"):
        CausalStep1TrajectoryDataset(
            _index([bad]),
            graph=_graph(),
            sensor_nodes=("N1",),
            history_steps=2,
            model_step_seconds=300,
        )


def test_step1_rejects_mixed_swmm_engines(tmp_path: Path) -> None:
    a = _trajectory(tmp_path, "a", [0, 300, 600], "5.2.4")
    b = _trajectory(tmp_path, "b", [0, 300, 600], "5.2.5")
    with pytest.raises(ValueError, match="mix SWMM engine"):
        CausalStep1TrajectoryDataset(
            _index([a, b]),
            graph=_graph(),
            sensor_nodes=("N1",),
            history_steps=2,
            model_step_seconds=300,
        )


def test_step1_accepts_t0_fixed_step_single_engine(tmp_path: Path) -> None:
    a = _trajectory(tmp_path, "a", [0, 300, 600], "5.2.4")
    dataset = CausalStep1TrajectoryDataset(
        _index([a]),
        graph=_graph(),
        sensor_nodes=("N1",),
        history_steps=2,
        model_step_seconds=300,
    )
    assert dataset.swmm_engine_version == "5.2.4"
    assert len(dataset) == 2
