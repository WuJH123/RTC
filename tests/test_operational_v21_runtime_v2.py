from __future__ import annotations

from pathlib import Path

import torch

import rtc.direct_tfv_operational_v21_runtime_v2 as runtime_v2
from rtc.direct_tfv_policy_return_query_margin_v17 import (
    LEGACY_V15_CHECKPOINT_CONTRACT,
    QueryConditionedPolicyReturnAdapterV17,
    rank_state_sha256,
)


def _v15_payload(adapter: QueryConditionedPolicyReturnAdapterV17) -> dict[str, object]:
    state: dict[str, torch.Tensor] = {}
    for key, value in adapter.state_dict().items():
        if key.startswith("rank_context_encoder."):
            state["context_encoder." + key[len("rank_context_encoder.") :]] = value.detach().clone()
        elif key.startswith("rank_candidate_encoder."):
            state["candidate_encoder." + key[len("rank_candidate_encoder.") :]] = value.detach().clone()
        elif key.startswith("rank_adjustment."):
            state[key] = value.detach().clone()
    return {
        "contract": LEGACY_V15_CHECKPOINT_CONTRACT,
        "base_step2_sha256": "a" * 64,
        "context_dim": adapter.context_dim,
        "candidate_dim": adapter.candidate_dim,
        "query_margin_state_dict": state,
        "validation_metrics": {
            "within_query_pairwise_rank_accuracy": 1.0,
            "within_query_candidate_top1_accuracy": 1.0,
        },
    }


def test_v15_rank_only_loader_accepts_selection_consistent_contract(tmp_path: Path) -> None:
    torch.manual_seed(7)
    source = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=1000.0,
        context_dim=13,
        candidate_dim=17,
    )
    checkpoint = tmp_path / "v15.pt"
    torch.save(_v15_payload(source), checkpoint)

    loaded, payload = runtime_v2.load_v15_rank_only_adapter(
        checkpoint,
        target_scale_m3=1000.0,
        base_step2_sha256="a" * 64,
        device=torch.device("cpu"),
    )

    assert payload["contract"] == LEGACY_V15_CHECKPOINT_CONTRACT
    assert rank_state_sha256(loaded) == rank_state_sha256(source)
    assert all(not parameter.requires_grad for parameter in loaded.rank_parameters())


def test_v15_rank_only_loader_rejects_wrong_step2_lineage(tmp_path: Path) -> None:
    source = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=1000.0,
        context_dim=13,
        candidate_dim=17,
    )
    checkpoint = tmp_path / "v15.pt"
    torch.save(_v15_payload(source), checkpoint)
    try:
        runtime_v2.load_v15_rank_only_adapter(
            checkpoint,
            target_scale_m3=1000.0,
            base_step2_sha256="b" * 64,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "Step2" in str(exc)
    else:
        raise AssertionError("V21 V2 accepted a V15 rank source from another Step2")


def test_v21_v2_rank_features_use_the_same_latent_builder_as_offline_training() -> None:
    assert runtime_v2.build_query_margin_v2_features.__module__.endswith(
        "direct_tfv_policy_return_query_margin_v2"
    )
    assert "build_query_margin_v2_features" in runtime_v2.DirectTFVOperationalV21MPCV2.optimize.__code__.co_names


def test_operational_runner_routes_to_v2_runtime() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_policy_direct_tfv_operational_v21_development.py"
    text = runner.read_text(encoding="utf-8")
    assert "direct_tfv_operational_v21_runtime_v2" in text
    assert "direct_tfv_operational_v21_runtime import" not in text
