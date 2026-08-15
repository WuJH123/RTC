from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtc.checkpoint_v128 import load_step2_v128, save_step2_v128
from rtc.development_profile_v128 import V128_EXECUTION_PROFILE_CONTRACT
from rtc.step2_differentiable_v128 import TypedActuatorMessageSurrogateV128
from rtc.step2_train_response_v60 import InputNormalizationV60


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        node_ids=("n0", "n1"),
        static_node_feature_names=("f0",),
        actuator_ids=("a0",),
        actuator_physics_feature_names=("min_setting", "max_setting"),
        system_units="SI",
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.zeros((2, 1), dtype=np.float32),
        actuator_upstream=np.asarray([0], dtype=np.int64),
        actuator_downstream=np.asarray([1], dtype=np.int64),
        actuator_physics=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )


def _normalization() -> InputNormalizationV60:
    return InputNormalizationV60(
        state_mean=np.zeros(3, dtype=np.float32),
        state_std=np.ones(3, dtype=np.float32),
        rainfall_mean=np.zeros(1, dtype=np.float32),
        rainfall_std=np.ones(1, dtype=np.float32),
        flow_mean=np.zeros(1, dtype=np.float32),
        flow_std=np.ones(1, dtype=np.float32),
    )


def _model() -> TypedActuatorMessageSurrogateV128:
    return TypedActuatorMessageSurrogateV128(
        state_dim=3,
        rainfall_dim=1,
        node_static_dim=1,
        actuator_physics_dim=2,
        actuator_count=1,
        hidden_dim=8,
        actuator_embedding_dim=2,
        action_message_dim=4,
        delta_state_scale=torch.ones(3),
        delta_flow_scale=torch.ones(1),
    )


def _current_checkpoint(tmp_path):
    graph = _graph()
    path = save_step2_v128(
        tmp_path / "current.pt",
        model=_model(),
        graph=graph,
        input_normalization=_normalization(),
        training_report={
            "contract": "test",
            "profile": "full",
            "execution_profile_contract": V128_EXECUTION_PROFILE_CONTRACT,
            "final_checkpoint_allowed": True,
        },
        lineage={"swmm_engine_version": "test"},
    )
    return graph, path


def test_v128_strict_saver_rejects_smoke_or_dev_artifacts(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit --profile full"):
        save_step2_v128(
            tmp_path / "smoke.pt",
            model=_model(),
            graph=_graph(),
            input_normalization=_normalization(),
            training_report={
                "profile": "smoke",
                "execution_profile_contract": V128_EXECUTION_PROFILE_CONTRACT,
                "final_checkpoint_allowed": False,
            },
            lineage={},
        )


def test_v128_loader_rejects_stale_model_source_fingerprint(tmp_path) -> None:
    graph, path = _current_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["v128_step2_source_sha256"] = "0" * 64
    stale = tmp_path / "stale_model.pt"
    torch.save(payload, stale)
    with pytest.raises(ValueError, match="model-source semantics changed"):
        load_step2_v128(stale, graph=graph, device="cpu")


def test_v128_loader_rejects_stale_training_source_fingerprint(tmp_path) -> None:
    graph, path = _current_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["v128_training_source_sha256"] = "0" * 64
    stale = tmp_path / "stale_training.pt"
    torch.save(payload, stale)
    with pytest.raises(ValueError, match="training-source semantics changed"):
        load_step2_v128(stale, graph=graph, device="cpu")


def test_v128_runtime_loader_rejects_nonfull_profile(tmp_path) -> None:
    graph, path = _current_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["execution_profile"] = "dev"
    bad = tmp_path / "dev_masquerading.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="rejects smoke/dev"):
        load_step2_v128(bad, graph=graph, device="cpu")
