from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from torch import nn

from rtc.direct_tfv_policy_return_facility_boundary_v20 import (
    FacilityBoundaryCalibratorV20,
    FacilityBoundaryPartsV20,
    ScaleOnlyPreprocessorV20,
    build_facility_boundary_parts_v20,
)


def _trainer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("train_step3_facility_boundary_v20_current")


class _FakeStep2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actuator_embedding = nn.Embedding(109, 4)
        with torch.no_grad():
            values = torch.arange(109 * 4, dtype=torch.float32).reshape(109, 4) / 100.0
            self.actuator_embedding.weight.copy_(values)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
    ) -> SimpleNamespace:
        del rainfall, previous_actuator_flow, actuator_upstream, actuator_downstream, actuator_physics
        delta = candidate_settings[:, 0, :] - reference_settings[:, 0, :]
        facility = delta * 100.0
        total = facility.sum(dim=1)
        interaction = torch.zeros_like(total)
        activity = torch.abs(delta)
        return SimpleNamespace(
            total_delta_tfv_m3=total,
            facility_main_effect_m3=facility,
            interaction_residual_m3=interaction,
            action_activity=activity,
        )


def _fixture() -> tuple[_FakeStep2, SimpleNamespace, SimpleNamespace, np.ndarray]:
    model = _FakeStep2()
    nodes = 120
    graph = SimpleNamespace(
        actuator_upstream=np.arange(109, dtype=np.int64) % nodes,
        actuator_downstream=(np.arange(109, dtype=np.int64) + 1) % nodes,
        actuator_physics=np.zeros((109, 2), dtype=np.float32),
    )
    normalization = SimpleNamespace(
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        rainfall_mean=np.zeros(1, dtype=np.float32),
        rainfall_std=np.ones(1, dtype=np.float32),
        flow_mean=np.zeros(109, dtype=np.float32),
        flow_std=np.ones(109, dtype=np.float32),
    )
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    return model, graph, normalization, mask


def _parts_for_changed_index(index: int) -> FacilityBoundaryPartsV20:
    model, graph, normalization, mask = _fixture()
    current_state = torch.arange(120 * 2, dtype=torch.float32).reshape(120, 2) / 10.0
    rainfall = torch.ones((1, 72, 120, 1), dtype=torch.float32)
    flow = torch.arange(109, dtype=torch.float32) / 100.0
    active = torch.zeros(109, dtype=torch.float32)
    candidate = active.clone()
    candidate[index] = 0.5
    return build_facility_boundary_parts_v20(
        step2_model=model,
        normalization=normalization,
        graph=graph,
        current_state=current_state,
        rainfall_scenarios=rainfall,
        previous_actuator_flow=flow,
        active_target=active,
        candidate_target=candidate,
        candidate_source="TYPE_AWARE_HYDRAULIC_PRESSURE",
        supervisory_mask=mask,
        target_scale_m3=1000.0,
    )


def test_v20_facility_identity_and_local_state_are_not_globally_pooled_away() -> None:
    first = _parts_for_changed_index(0).feature
    second = _parts_for_changed_index(10).feature
    assert tuple(first.shape) == tuple(second.shape)
    assert not torch.allclose(first, second)


def test_v20_zero_feature_has_exact_zero_boundary_without_intercept() -> None:
    width = int(_parts_for_changed_index(0).feature.numel())
    preprocessor = ScaleOnlyPreprocessorV20(feature_scale=torch.ones(width))
    calibrator = FacilityBoundaryCalibratorV20(
        preprocessor=preprocessor,
        boundary_weight=torch.linspace(-1.0, 1.0, width),
        magnitude_weight=torch.linspace(1.0, 2.0, width),
        target_scale_m3=1000.0,
    )
    output = calibrator.predict(FacilityBoundaryPartsV20(feature=torch.zeros(width)))
    assert float(output.hold_score) == 0.0
    assert float(output.advantage_m3) == 0.0
    assert bool(output.execute) is False


def test_v20_no_intercept_logistic_preserves_zero_score() -> None:
    trainer = _trainer()
    z = np.asarray(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-2.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    returns = np.asarray([5.0, 3.0, -5.0, -3.0], dtype=float)
    queries = np.asarray(["H1", "H2", "A1", "A2"])
    weight = trainer._fit_logistic_no_intercept(z, returns, queries, ridge=0.1)
    assert weight.shape == (2,)
    assert float(np.zeros(2) @ weight) == 0.0


def test_v20_group_oof_keeps_sibling_candidates_together() -> None:
    trainer = _trainer()
    query_count = 12
    records_per_query = 2
    n = query_count * records_per_query
    queries = np.asarray(
        [f"Q{query:02d}" for query in range(query_count) for _ in range(records_per_query)]
    )
    selected_mask = np.asarray(
        [local == 0 for _query in range(query_count) for local in range(records_per_query)],
        dtype=bool,
    )
    returns: list[float] = []
    features = np.zeros((n, 5), dtype=float)
    for query in range(query_count):
        selected_return = 4.0 + query if query < 6 else -(4.0 + query)
        sibling_return = selected_return * 0.5
        returns.extend((selected_return, sibling_return))
        sign = 1.0 if selected_return >= 0.0 else -1.0
        for local in range(records_per_query):
            row = query * records_per_query + local
            features[row] = [sign, sign * 0.5, query / 20.0, local * 0.1, sign * query / 20.0]
    rows = {
        "features": features,
        "returns": np.asarray(returns, dtype=float),
        "queries": queries,
        "sources": np.asarray(["TYPE_AWARE_HYDRAULIC_PRESSURE"] * n),
        "selected_mask": selected_mask,
        "rank_scores": np.tile(np.asarray([0.0, 1.0]), query_count),
    }
    result = trainer._crossfit(rows, seed=42)
    assert result["query_group_disjoint"] is True
    assert len(result["fold_query_counts"]) == 6
    assert sum(result["fold_query_counts"]) == query_count
    assert result["selected"]["ridge"] in trainer.RIDGE_GRID
    assert np.isfinite(result["selected"]["selected_auc"])
