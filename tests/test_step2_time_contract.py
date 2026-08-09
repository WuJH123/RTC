from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rtc.dataset_compile import compile_branches_to_npz


def _branch(root: Path, name: str, step: int, engine: str = "5.2.4") -> Path:
    node_ids = np.asarray(["N1"])
    actuator_ids = np.asarray(["A1"])
    times = np.asarray([0, step, 2 * step], dtype=np.int64)
    compact = root / f"{name}.npz"
    np.savez_compressed(
        compact,
        elapsed_seconds=times,
        node_ids=node_ids,
        actuator_ids=actuator_ids,
        state_si=np.zeros((3, 1, 6), dtype=np.float32),
        rainfall_mmhr=np.zeros((3, 1, 1), dtype=np.float32),
        actuator_flow_m3s=np.zeros((3, 1), dtype=np.float32),
        candidate_setting=np.asarray([0.5], dtype=np.float32),
    )
    stats = root / f"{name}.stats.csv"
    stats.write_text("node_id,delta_flooding_volume_m3\nN1,0\n", encoding="utf-8")
    meta = root / f"{name}.json"
    meta.write_text(
        json.dumps(
            {
                "compact_file": compact.name,
                "node_statistics_file": stats.name,
                "candidate_action_sha256": name,
                "swmm_engine_version": engine,
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_step2_dataset_rejects_phase0_and_production_step_mixture(tmp_path: Path) -> None:
    fast = _branch(tmp_path, "fast60", 60)
    production = _branch(tmp_path, "production300", 300)
    with pytest.raises(ValueError, match="time-step mismatch"):
        compile_branches_to_npz([fast, production], tmp_path / "mixed.npz")


def test_step2_dataset_rejects_mixed_swmm_engines(tmp_path: Path) -> None:
    a = _branch(tmp_path, "a", 300, engine="5.2.4")
    b = _branch(tmp_path, "b", 300, engine="5.2.5")
    with pytest.raises(ValueError, match="different SWMM engine"):
        compile_branches_to_npz([a, b], tmp_path / "mixed_engine.npz")


def test_step2_dataset_embeds_frozen_time_and_engine_contract(tmp_path: Path) -> None:
    a = _branch(tmp_path, "a", 300)
    b = _branch(tmp_path, "b", 300)
    out = compile_branches_to_npz([a, b], tmp_path / "production.npz")
    with np.load(out, allow_pickle=False) as raw:
        assert int(raw["model_step_seconds"].item()) == 300
        assert int(raw["horizon_steps"].item()) == 2
        assert str(raw["swmm_engine_version"].item()) == "5.2.4"
