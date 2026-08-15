from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.models import SparseStateEstimator
from rtc.step1_runtime_v127 import load_frozen_step1_v127


def _artifacts(tmp_path: Path, *, mismatch: bool) -> Path:
    actuator_ids = np.asarray(["a0", "a1"])
    compact = tmp_path / "run.npz"
    targets = np.asarray([[0.5, 0.5], [0.8, 0.2]], dtype=np.float32)
    if mismatch:
        targets[1, 0] = 0.7
    np.savez_compressed(
        compact,
        elapsed_seconds=np.asarray([0, 600], dtype=np.int64),
        actuator_ids=actuator_ids,
        target_setting=targets,
    )
    decision = tmp_path / "run.decisions.jsonl"
    decision.write_text(
        json.dumps(
            {
                "elapsed_seconds": 600,
                "settings": {"a0": 0.8, "a1": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    meta = tmp_path / "run.json"
    meta.write_text(
        json.dumps(
            {
                "compact_file": compact.name,
                "decision_file": decision.name,
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_target_write_audit_passes_same_epoch_readback(tmp_path: Path) -> None:
    result = audit_target_write_readback_v127(metadata_path=_artifacts(tmp_path, mismatch=False))
    assert result["passed"] is True
    assert result["decision_count"] == 1
    assert float(result["max_target_write_readback_error"]) < 1e-6
    assert result["current_setting_used_as_write_acceptance"] is False


def test_target_write_audit_rejects_same_epoch_mismatch(tmp_path: Path) -> None:
    result = audit_target_write_readback_v127(metadata_path=_artifacts(tmp_path, mismatch=True))
    assert result["passed"] is False
    assert result["failed_decisions"] == 1
    assert float(result["max_target_write_readback_error"]) > 0.05


def test_v127_frozen_step1_loader_does_not_require_current_project_source_hash(
    tmp_path: Path,
) -> None:
    model = SparseStateEstimator(
        observed_dim=2,
        static_dim=3,
        state_dim=4,
        hidden_dim=8,
        graph_layers=1,
        context_dim=1,
        history_steps=13,
        model_step_seconds=300,
        swmm_engine_version="test-engine",
        context_contract="test-context",
    )
    path = tmp_path / "step1.pt"
    torch.save(
        {
            "checkpoint_contract": "RTC_TORCH_CHECKPOINT_V2_CODE_BOUND",
            "scientific_split": "development",
            # Deliberately unrelated to the current project implementation contract.
            "rtc_source_tree_sha256": "0" * 64,
            "model_config": {
                "observed_dim": 2,
                "static_dim": 3,
                "state_dim": 4,
                "hidden_dim": 8,
                "graph_layers": 1,
                "context_dim": 1,
                "history_steps": 13,
                "model_step_seconds": 300,
                "swmm_engine_version": "test-engine",
                "context_contract": "test-context",
                "training_contract_sha256": "training-provenance",
            },
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = load_frozen_step1_v127(path, "cpu")
    assert isinstance(loaded, SparseStateEstimator)
    assert loaded.runtime_metadata["model_step_seconds"] == 300
    assert loaded.runtime_metadata["original_rtc_source_tree_sha256"] == "0" * 64
