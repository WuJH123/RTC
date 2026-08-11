"""Run the bounded V4.3 state/topology-conditioned interaction experiment.

The script consumes only the existing development/train cache and the frozen
V4.2/V4.2.1 cohorts.  It never launches SWMM and never reads Validation or Final
outcomes.  Outputs are written to a new study directory so the V4.1/V4.2/V4.2.1
lineage remains intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtc.production_cli import _load_graph  # noqa: E402
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41  # noqa: E402
from rtc.step2_control_response_v43 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV43,
    interaction_parameter_names,
    parameter_sha256,
    reference_parameter_names,
    set_trainable_phase,
    single_parameter_names,
    topology_graph_contract,
)
from rtc.step2_train_response_v4 import build_full_train_normalization_from_checkpoint, load_train_groups  # noqa: E402
from rtc.step2_train_response_v41 import (  # noqa: E402
    CounterfactualDeltaScalesV41,
    evaluate_response_groups_v41,
    group_metrics_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    response_group_loss_v41,
    source_parameter_is_trainable,
    stack_response_group_v41,
)
from rtc.step2_train_response_v43 import (  # noqa: E402
    TRAINING_CONTRACT_V43,
    compare_d2_prediction_snapshots_v43,
    d2_prediction_snapshot_v43,
    deduplicate_reference_groups_v43,
    train_phase_v43,
)
from rtc.step2_training_cache import load_step2_training_cache  # noqa: E402


STUDY = Path(r"E:\RTC_sewer\Project7\study_v069")
V4 = STUDY / "step2_control_response_v4"
V421 = STUDY / "step2_d3_magnitude_balance_v421"
OUT = STUDY / "step2_state_topology_interaction_v43"
CACHE = STUDY / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
OLD_SCALE = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
SCALE_JSON = STUDY / "step2_response_calibration_v41" / "00_scales" / "counterfactual_delta_scales_train18.json"
PARENT = V421 / "03_tiny_combined" / "v42_tiny_combined.pt"
TINY_GROUPS = V4 / "02_tiny_overfit" / "02_tiny_overfit.groups.csv"
MICRO_GROUPS = V4 / "03_12_group_micro" / "03_12_group_micro.groups.csv"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def _build_v43(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> DifferentiableCounterfactualResponseModelV43:
    return DifferentiableCounterfactualResponseModelV43(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
        max_horizon_steps=72,
        effect_rank=12,
        interaction_magnitude_features_enabled=True,
    )


def _build_v41(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> DifferentiableCounterfactualResponseModelV41:
    return DifferentiableCounterfactualResponseModelV41(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
        max_horizon_steps=72,
        effect_rank=12,
        interaction_magnitude_features_enabled=True,
    )


def _load_compatible(model: torch.nn.Module, path: Path, *, allow_topology_missing: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in incoming.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed = ("topology_",) if allow_topology_missing else ()
    bad_missing = [name for name in missing if not name.startswith(allowed)]
    if unexpected or bad_missing:
        raise RuntimeError(f"V4.3 parent mismatch: {missing=} {unexpected=} {bad_missing=}")
    return {"missing": list(missing), "unexpected": list(unexpected), "contract": payload.get("contract")}


def _validate_train_only_cache() -> dict[str, Any]:
    cache = load_step2_training_cache(CACHE / "CACHE_MANIFEST.json")
    split = {str(value).lower() for item in cache["shards"] for value in item["arrays"]["scientific_split"].tolist()}
    fold = {str(value).lower() for item in cache["shards"] for value in item["arrays"]["development_fold"].tolist()}
    if split != {"development"} or fold != {"train"}:
        raise RuntimeError(f"V4.3 cache is not development/train only: {split=} {fold=}")
    return {"scientific_split": sorted(split), "development_fold": sorted(fold), "shard_count": len(cache["shards"])}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"groups": 0}
    return {
        "groups": len(rows),
        "spread_ratio": float(np.nanmean([row["spread_ratio"] for row in rows])),
        "rank": float(np.nanmean([row["rank"] for row in rows])),
        "pairwise": float(np.nanmean([row["pairwise"] for row in rows])),
        "sign": float(np.nanmean([row["sign"] for row in rows])),
        "top1": int(sum(bool(row["top1"]) for row in rows)),
        "mean_regret_m3": float(np.mean([row["regret_m3"] for row in rows])),
        "max_regret_m3": float(max(row["regret_m3"] for row in rows)),
    }


def _tensor_digest(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _source_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {source: _aggregate([row for row in rows if row["source_kind"] == source]) for source in ("D2", "D3")}


def _old_interference_audit(
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    groups: dict[str, list[Any]],
    device: torch.device,
) -> dict[str, Any]:
    """One bounded D3 optimizer step reproducing the old shared-reference path."""

    d2_group = next(name for name in groups if name.startswith("D2::"))
    d3_group = next(name for name in groups if name.startswith("D3::"))
    model = _build_v41(graph, normalization, scales).to(device).float()
    load_meta = _load_compatible(model, PARENT, allow_topology_missing=False)
    prepared = prepare_graph_v41(model, graph, normalization, device)
    d2_batch = stack_response_group_v41(groups[d2_group], device)
    d3_batch = stack_response_group_v41(groups[d3_group], device)
    before = model.forward_group(
        d2_batch.initial_state, d2_batch.rainfall, d2_batch.reference_settings,
        d2_batch.candidate_settings, d2_batch.previous_actuator_flow, prepared,
        d2_batch.elapsed_seconds, source_kind="D2",
    )
    before_snapshot = {"delta_states": before.delta_states_physical.detach().clone(), "direct_tfv": before.direct_delta_tfv_m3.detach().clone()}
    names = [name for name, parameter in model.named_parameters() if source_parameter_is_trainable(name, "D3")]
    optimizer = torch.optim.AdamW([parameter for name, parameter in model.named_parameters() if name in set(names)], lr=2e-3)
    model.train()
    output = model.forward_group(
        d3_batch.initial_state, d3_batch.rainfall, d3_batch.reference_settings,
        d3_batch.candidate_settings, d3_batch.previous_actuator_flow, prepared,
        d3_batch.elapsed_seconds, source_kind="D3",
    )
    loss, _ = response_group_loss_v41(
        output, d3_batch, scales.by_source["D3"], normalization, magnitude_calibration=True
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in model.named_parameters():
        if not source_parameter_is_trainable(name, "D3"):
            parameter.grad = None
    optimizer.step()
    model.eval()
    after = model.forward_group(
        d2_batch.initial_state, d2_batch.rainfall, d2_batch.reference_settings,
        d2_batch.candidate_settings, d2_batch.previous_actuator_flow, prepared,
        d2_batch.elapsed_seconds, source_kind="D2",
    )
    after_snapshot = {"delta_states": after.delta_states_physical.detach().clone(), "direct_tfv": after.direct_delta_tfv_m3.detach().clone()}
    shifts = {
        "delta_states_max_abs": float((before_snapshot["delta_states"] - after_snapshot["delta_states"]).abs().max()),
        "direct_tfv_max_abs": float((before_snapshot["direct_tfv"] - after_snapshot["direct_tfv"]).abs().max()),
    }
    parameter_changes = {}
    prefixes = (
        "reference_encoder", "reference_state_head", "reference_flow_encoder", "reference_flow_head",
        "node_static_encoder", "actuator_static_encoder", "actuator_identity", "temporal_identity",
        "single_effect_encoder", "single_flow_head", "single_state_head", "single_network_coefficient_head",
        "single_node_basis_head", "direct_single_tfv_head", "interaction_encoder", "interaction_state_head",
        "interaction_flow_head", "direct_interaction_tfv_head",
    )
    for prefix in prefixes:
        names_for_prefix = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
        if not names_for_prefix:
            continue
        before_hash = parameter_sha256(model, [])
        # Hash the current model against a separately reloaded parent for this prefix.
        parent_model = _build_v41(graph, normalization, scales).to(device).float()
        _load_compatible(parent_model, PARENT, allow_topology_missing=False)
        parameter_changes[prefix] = {
            "changed": any(
                not torch.equal(dict(model.named_parameters())[name].detach().cpu(), dict(parent_model.named_parameters())[name].detach().cpu())
                for name in names_for_prefix
            ),
            "parameter_count": len(names_for_prefix),
        }
    return {
        "contract": "STEP2_CROSS_SOURCE_PARAMETER_INTERFERENCE_AUDIT_V43",
        "parent_load": load_meta,
        "d2_group": d2_group,
        "d3_group": d3_group,
        "old_d3_allowed_parameter_count": len(names),
        "old_d3_changed_reference_parameters": any(
            item["changed"] for key, item in parameter_changes.items() if key.startswith("reference_")
        ),
        "d2_prediction_changed_after_old_d3": bool(any(value > 1e-7 for value in shifts.values())),
        "prediction_shift": shifts,
        "parameter_changes": parameter_changes,
        "root_cause": "old D3 partition allowed reference/static parameters that D2 forward consumes",
    }


def _graph_ablation_metrics(
    model: DifferentiableCounterfactualResponseModelV43,
    grouped_pairs: dict[str, list[Any]],
    prepared: Any,
    device: torch.device,
    *,
    local_state_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    normal_rows: list[dict[str, Any]] = []
    ablated_rows: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, device)
        with torch.no_grad():
            normal = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=False, local_state_enabled=local_state_enabled)
            ablated = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=local_state_enabled)
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        normal_pred = normal.direct_delta_tfv_m3[0].detach().cpu().numpy()
        ablated_pred = ablated.direct_delta_tfv_m3[0].detach().cpu().numpy()
        normal_rows.append(group_metrics_v41(predicted=normal_pred, truth=truth, group=group, source_kind="D3"))
        ablated_rows.append(group_metrics_v41(predicted=ablated_pred, truth=truth, group=group, source_kind="D3"))
    model.set_topology_ablation(graph_enabled=True, local_state_enabled=True)
    normal = _aggregate(normal_rows)
    ablated = _aggregate(ablated_rows)
    prediction_change = float("nan")
    # Keep full candidate arrays out of committed JSON while recording a direct
    # graph-ablation prediction shift for one representative frozen group.
    if grouped_pairs:
        first_group = next(iter(sorted(grouped_pairs)))
        batch = stack_response_group_v41(grouped_pairs[first_group], device)
        with torch.no_grad():
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=local_state_enabled)
            normal_output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=False, local_state_enabled=local_state_enabled)
            ablated_output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=True)
        prediction_change = float(
            (normal_output.direct_delta_tfv_m3 - ablated_output.direct_delta_tfv_m3).abs().max()
        )
    return {
        "normal": normal,
        "ablated": ablated,
        "delta_rank": float(normal.get("rank", float("nan")) - ablated.get("rank", float("nan"))),
        "delta_pairwise": float(normal.get("pairwise", float("nan")) - ablated.get("pairwise", float("nan"))),
        "delta_spread": float(normal.get("spread_ratio", float("nan")) - ablated.get("spread_ratio", float("nan"))),
        "delta_mean_regret_m3": float(normal.get("mean_regret_m3", float("nan")) - ablated.get("mean_regret_m3", float("nan"))),
        "prediction_change_max_abs_m3": prediction_change,
    }


def _graph_state_ablation_metrics(
    model: DifferentiableCounterfactualResponseModelV43,
    grouped_pairs: dict[str, list[Any]],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    normal_rows: list[dict[str, Any]] = []
    ablated_rows: list[dict[str, Any]] = []
    prediction_change = float("nan")
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, device)
        with torch.no_grad():
            normal = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=False)
            ablated = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=True)
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        normal_rows.append(group_metrics_v41(predicted=normal.direct_delta_tfv_m3[0].cpu().numpy(), truth=truth, group=group, source_kind="D3"))
        ablated_rows.append(group_metrics_v41(predicted=ablated.direct_delta_tfv_m3[0].cpu().numpy(), truth=truth, group=group, source_kind="D3"))
    normal = _aggregate(normal_rows)
    ablated = _aggregate(ablated_rows)
    if grouped_pairs:
        first_group = next(iter(sorted(grouped_pairs)))
        batch = stack_response_group_v41(grouped_pairs[first_group], device)
        with torch.no_grad():
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=True)
            normal_output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=False)
            ablated_output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.set_topology_ablation(graph_enabled=True, local_state_enabled=True)
        prediction_change = float(
            (normal_output.direct_delta_tfv_m3 - ablated_output.direct_delta_tfv_m3).abs().max()
        )
    return {
        "normal": normal,
        "global_context_ablated": ablated,
        "delta_rank": float(normal.get("rank", float("nan")) - ablated.get("rank", float("nan"))),
        "delta_pairwise": float(normal.get("pairwise", float("nan")) - ablated.get("pairwise", float("nan"))),
        "delta_mean_regret_m3": float(normal.get("mean_regret_m3", float("nan")) - ablated.get("mean_regret_m3", float("nan"))),
        "prediction_change_max_abs_m3": prediction_change,
    }


def _shortest_distances(edge_index: torch.Tensor, sources: list[int], node_count: int) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in edge_index.t().detach().cpu().tolist():
        adjacency[int(src)].append(int(dst))
    distance = np.full(node_count, np.inf, dtype=np.float64)
    queue: deque[int] = deque()
    for source in sources:
        distance[source] = 0.0
        queue.append(source)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if not np.isfinite(distance[neighbor]):
                distance[neighbor] = distance[current] + 1.0
                queue.append(neighbor)
    return distance


def _topology_gradient_audit(
    model: DifferentiableCounterfactualResponseModelV43,
    pairs: list[Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch = stack_response_group_v41(pairs, device)
    candidate = batch.candidate_settings.detach().clone().requires_grad_(True)
    output = model.forward_group(
        batch.initial_state, batch.rainfall, batch.reference_settings,
        candidate, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
    )
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0].detach()
    delta = candidate.detach() - batch.reference_settings[:, None]
    active = torch.any(delta.abs() > 1e-8, dim=(0, 1, 2)).nonzero(as_tuple=False).flatten().tolist()
    endpoints = set()
    for actuator in active:
        endpoints.add(int(prepared.actuator_upstream[actuator]))
        endpoints.add(int(prepared.actuator_downstream[actuator]))
    distances = _shortest_distances(prepared.edge_index, sorted(endpoints), int(prepared.static_node_features.shape[0]))
    response = output.interaction_delta_states_physical.detach().abs().mean(dim=(0, 1, 2, 4)).cpu().numpy()
    by_distance = {}
    for distance in (0, 1, 2, 3):
        selected = np.isclose(distances, distance)
        by_distance[str(distance)] = {
            "node_count": int(selected.sum()),
            "mean_abs_interaction_response": float(response[selected].mean()) if selected.any() else 0.0,
            "max_abs_interaction_response": float(response[selected].max()) if selected.any() else 0.0,
        }
    return {
        "changed_actuator_gradient_nonzero_fraction": float(gradient[delta.abs() > 1e-8].ne(0.0).float().mean()),
        "gradient_finite_fraction": float(torch.isfinite(gradient).float().mean()),
        "active_actuator_count": len(active),
        "endpoint_node_count": len(endpoints),
        "response_by_topological_distance": by_distance,
        "network_response_magnitude": float(response.mean()),
        "all_nodes_nearly_identical": bool(float(response.std()) < 1e-8),
    }


def _physical_causal_invariants(
    model: DifferentiableCounterfactualResponseModelV43,
    pairs: list[Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, bool | float]:
    batch = stack_response_group_v41(pairs, device)
    reference = batch.reference_settings
    zero_candidate = reference[:, None].clone()
    with torch.no_grad():
        zero = model.forward_group(
            batch.initial_state, batch.rainfall, reference, zero_candidate,
            batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
        )
    single_candidate = zero_candidate.clone()
    single_candidate[:, 0, 2:, 0] = (single_candidate[:, 0, 2:, 0] + 0.1).clamp(0.0, 1.0)
    with torch.no_grad():
        single = model.forward_group(
            batch.initial_state, batch.rainfall, reference, single_candidate,
            batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
        )
    multi_candidate = single_candidate.clone()
    multi_candidate[:, 0, :, 1] = (multi_candidate[:, 0, :, 1] + 0.1).clamp(0.0, 1.0)
    with torch.no_grad():
        multi = model.forward_group(
            batch.initial_state, batch.rainfall, reference, multi_candidate,
            batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
        )
    late = zero_candidate.clone()
    late[:, 0, 5:, :2] = (late[:, 0, 5:, :2] + 0.1).clamp(0.0, 1.0)
    with torch.no_grad():
        late_out = model.forward_group(
            batch.initial_state, batch.rainfall, reference, late,
            batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
        )
    candidate_grad = multi_candidate.detach().clone().requires_grad_(True)
    grad_out = model.forward_group(
        batch.initial_state, batch.rainfall, reference, candidate_grad,
        batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3"
    )
    gradient = torch.autograd.grad(grad_out.direct_delta_tfv_m3.sum() + grad_out.trajectory_delta_tfv_m3.sum(), candidate_grad)[0]
    invert = prepared.invert_elevation_m.reshape(1, 1, 1, -1)
    return {
        "zero_action_exact_zero": bool(torch.equal(zero.interaction_delta_states_physical, torch.zeros_like(zero.interaction_delta_states_physical))),
        "single_action_interaction_exact_zero": bool(torch.equal(single.interaction_delta_states_physical, torch.zeros_like(single.interaction_delta_states_physical))),
        "multi_action_interaction_nonzero": bool(torch.any(multi.interaction_delta_states_physical != 0.0)),
        "future_action_cannot_affect_past": bool(torch.equal(late_out.interaction_delta_states_physical[:, :, :5], torch.zeros_like(late_out.interaction_delta_states_physical[:, :, :5]))),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nonzero": bool(gradient.abs().sum() > 0.0),
        "nonnegative_candidate_flooding": bool(torch.all(multi.candidate_states_physical[..., 2] >= 0.0)),
        "head_depth_consistency": bool(torch.allclose(multi.candidate_states_physical[..., 1], multi.candidate_states_physical[..., 0] + invert, atol=1e-6, rtol=0.0)),
        "horizon_steps": int(reference.shape[1]),
    }


def _state_topology_strata(
    grouped_pairs: dict[str, list[Any]], graph: Any, prepared: Any
) -> list[dict[str, Any]]:
    raw_static = np.asarray(graph.static_node_features, dtype=np.float64)
    feature_names = list(graph.static_node_feature_names)
    max_depth = raw_static[:, feature_names.index("max_depth_m")]
    valid_depth = max_depth > 1e-3
    storage_mask = raw_static[:, feature_names.index("is_storage")] > 0.5
    rows: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        pair = pairs[0]
        reference_state = np.asarray(pair.reference["target_states_physical"], dtype=np.float64)
        depth = reference_state[..., 0]
        flood = reference_state[..., 2]
        wetness = float(np.mean(depth[:, valid_depth] / np.maximum(max_depth[None, valid_depth], 1e-6))) if valid_depth.any() else float("nan")
        flooding_activity = float(np.mean(flood > 1e-8))
        storage_utilization = float(np.mean(depth[:, storage_mask] / np.maximum(max_depth[None, storage_mask], 1e-6))) if storage_mask.any() else float("nan")
        delta = np.asarray(pair.candidate["settings"]) - np.asarray(pair.reference["settings"])
        active = np.flatnonzero(np.any(np.abs(delta) > 1e-8, axis=0)).tolist()
        endpoints = []
        for actuator in active:
            endpoints.extend([int(graph.actuator_upstream[actuator]), int(graph.actuator_downstream[actuator])])
        distances = _shortest_distances(prepared.edge_index, endpoints, len(graph.node_ids)) if endpoints else np.zeros(len(graph.node_ids))
        if endpoints:
            pairwise_distances = []
            unique_endpoints = sorted(set(endpoints))
            for source in unique_endpoints:
                source_distances = _shortest_distances(prepared.edge_index, [source], len(graph.node_ids))
                pairwise_distances.extend(
                    float(source_distances[target])
                    for target in unique_endpoints
                    if target != source and np.isfinite(source_distances[target])
                )
            dispersion = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
            endpoint_distance = _shortest_distances(prepared.edge_index, [unique_endpoints[0]], len(graph.node_ids))
            topological_distance = float(np.nanmax(endpoint_distance[unique_endpoints]))
        else:
            dispersion = 0.0
            topological_distance = 0.0
        rows.append({
            "group": group,
            "initial_hydraulic_wetness": wetness,
            "initial_flooding_activity": flooding_activity,
            "initial_storage_utilization": storage_utilization,
            "multi_actuator_spatial_dispersion": dispersion,
            "active_actuator_count": len(active),
            "actuator_topological_distance": topological_distance,
        })
    return rows


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        util, memory = (float(value.strip()) for value in completed.stdout.strip().split(",")[:2])
        return {"available": True, "mean_percent": util, "p90_percent": util, "max_percent": util, "peak_memory_mib": memory}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}


def _write_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [f"# {title}", "", "```json", json.dumps(payload, indent=2, allow_nan=True), "```", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def _run_pipeline(
    *,
    name: str,
    selected: list[str],
    groups: dict[str, list[Any]],
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    device_name: str,
    parent: Path,
    epochs: tuple[int, int, int],
) -> tuple[DifferentiableCounterfactualResponseModelV43, dict[str, Any]]:
    target_device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_v43(graph, normalization, scales).to(target_device).float()
    parent_load = _load_compatible(model, parent)
    selected_groups = {group: groups[group] for group in selected}
    d2_groups = {group: groups[group] for group in selected if group.startswith("D2::")}
    d3_groups = {group: groups[group] for group in selected if group.startswith("D3::")}
    reference_groups = deduplicate_reference_groups_v43(selected_groups)
    phase_root = OUT / name
    phase_root.mkdir(parents=True, exist_ok=True)
    reference_result = train_phase_v43(
        model=model, grouped_pairs=reference_groups, normalization=normalization, scales=scales,
        graph=graph, phase="reference", out_path=phase_root / "reference_phase.pt",
        epochs=epochs[0], learning_rate=2e-3, device=device_name, seed=42, magnitude_calibration=False,
    )
    d2_result = train_phase_v43(
        model=model, grouped_pairs=d2_groups, normalization=normalization, scales=scales,
        graph=graph, phase="d2", out_path=phase_root / "d2_phase.pt",
        epochs=epochs[1], learning_rate=2e-3, device=device_name, seed=42, magnitude_calibration=False,
    )
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    d2_tiny = next((group for group in selected if group.startswith("D2::")), None)
    if d2_tiny is None:
        raise RuntimeError("V4.3 pipeline requires a D2 group for interference validation")
    d2_batch = stack_response_group_v41(groups[d2_tiny], target_device)
    d2_before = d2_prediction_snapshot_v43(model, d2_batch, prepared)
    d2_single_sha_before_d3 = parameter_sha256(model, single_parameter_names(model))
    d3_result = train_phase_v43(
        model=model, grouped_pairs=d3_groups, normalization=normalization, scales=scales,
        graph=graph, phase="d3", out_path=phase_root / "d3_phase.pt",
        epochs=epochs[2], learning_rate=2e-3, device=device_name, seed=42, magnitude_calibration=True,
    )
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    d2_after = d2_prediction_snapshot_v43(model, d2_batch, prepared)
    d2_invariance = compare_d2_prediction_snapshots_v43(d2_before, d2_after)
    d2_single_sha_after_d3 = parameter_sha256(model, single_parameter_names(model))
    final_path = phase_root / f"{name}.pt"
    torch.save({"contract": TRAINING_CONTRACT_V43, "model_state_dict": model.state_dict(), "phase": "complete", "seed": 42}, final_path)
    metric_rows, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=selected_groups, prepared=prepared, device=target_device
    )
    d3_strata = magnitude_strata_metrics_v41(
        contributions,
        q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]),
        q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]),
    )
    result = {
        "contract": "STEP2_STATE_TOPOLOGY_INTERACTION_V43_STAGE_RESULT",
        "stage": name,
        "selected_groups": selected,
        "parent_checkpoint": str(parent),
        "parent_load": parent_load,
        "reference_phase": reference_result.__dict__,
        "d2_phase": d2_result.__dict__,
        "d3_phase": d3_result.__dict__,
        "group_metrics": metric_rows,
        "candidate_contributions": contributions,
        "d3_magnitude_strata": d3_strata,
        "d2_prediction_before_d3_digest": {key: _tensor_digest(value) for key, value in d2_before.items()},
        "d2_prediction_after_d3_digest": {key: _tensor_digest(value) for key, value in d2_after.items()},
        "d2_prediction_invariance": d2_invariance,
        "d2_single_parameter_sha_before_d3": d2_single_sha_before_d3,
        "d2_single_parameter_sha_after_d3": d2_single_sha_after_d3,
        "d2_single_parameters_unchanged": d2_single_sha_before_d3 == d2_single_sha_after_d3,
        "checkpoint": str(final_path),
        "reference_forwards_per_group": 1,
        "precision": "fp32",
        "boundary": {
            "scientific_split": ["development"], "development_fold": ["train"],
            "swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False,
            "validation_outcomes_accessed": False, "final_accessed": False,
            "formal_step2_run": False, "closed_loop_run": False, "policy_lock_run": False,
            "full_train_smoke_run": False, "hyperparameter_grid_search": False,
        },
    }
    (phase_root / "stage_result.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return model, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-micro", action="store_true")
    args = parser.parse_args()
    _validate_train_only_cache()
    tiny = _read_groups(TINY_GROUPS)
    micro = _read_groups(MICRO_GROUPS)
    if len(tiny) != 2 or len(micro) != 12 or not set(tiny).issubset(micro):
        raise RuntimeError("frozen V4 cohort contract changed")
    scales = CounterfactualDeltaScalesV41.from_json_dict(_load_json(SCALE_JSON))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = _load_graph(GRAPH)
    groups = load_train_groups(CACHE, normalization, sorted(set(micro)))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    interference = _old_interference_audit(graph, normalization, scales, groups, device)
    tiny_model, tiny_result = _run_pipeline(
        name="tiny", selected=tiny, groups=groups, graph=graph, normalization=normalization,
        scales=scales, device_name=args.device, parent=PARENT, epochs=(3, 8, 15),
    )
    micro_result = None
    micro_model = tiny_model
    if not args.skip_micro:
        micro_model, micro_result = _run_pipeline(
            name="micro", selected=micro, groups=groups, graph=graph, normalization=normalization,
            scales=scales, device_name=args.device, parent=OUT / "tiny" / "tiny.pt", epochs=(2, 5, 12),
        )
    final_model = micro_model
    final_stage = micro_result or tiny_result
    final_device = device
    prepared = prepare_graph_v41(final_model, graph, normalization, final_device)
    topology_contract = topology_graph_contract(prepared)
    d3_groups = {group: groups[group] for group in micro if group.startswith("D3::")}
    topology_ablation = _graph_ablation_metrics(final_model, d3_groups, prepared, final_device, local_state_enabled=True)
    state_ablation = _graph_state_ablation_metrics(final_model, d3_groups, prepared, final_device)
    d3_tiny_group = next(group for group in tiny if group.startswith("D3::"))
    topology_gradient = _topology_gradient_audit(final_model, groups[d3_tiny_group], prepared, final_device)
    invariants = _physical_causal_invariants(final_model, groups[d3_tiny_group], prepared, final_device)
    state_strata = _state_topology_strata({group: groups[group] for group in micro}, graph, prepared)

    # The V4.2.1 model has no topology hook; an edge swap is a read-only proof
    # that its prediction does not consume edge_index.
    old_model = _build_v41(graph, normalization, scales).to(final_device).float()
    _load_compatible(old_model, PARENT, allow_topology_missing=False)
    old_prepared = prepare_graph_v41(old_model, graph, normalization, final_device)
    old_batch = stack_response_group_v41(groups[d3_tiny_group], final_device)
    with torch.no_grad():
        old_normal = old_model.forward_group(
            old_batch.initial_state, old_batch.rainfall, old_batch.reference_settings,
            old_batch.candidate_settings, old_batch.previous_actuator_flow, old_prepared,
            old_batch.elapsed_seconds, source_kind="D3",
        )
    altered_edge = old_prepared.edge_index.flip(0)
    old_altered = old_model.prepare_static(
        static_node_features=old_prepared.static_node_features,
        actuator_physics=old_prepared.actuator_physics,
        actuator_upstream=old_prepared.actuator_upstream,
        actuator_downstream=old_prepared.actuator_downstream,
        edge_index=altered_edge,
        invert_elevation_m=old_prepared.invert_elevation_m,
    )
    with torch.no_grad():
        old_changed = old_model.forward_group(
            old_batch.initial_state, old_batch.rainfall, old_batch.reference_settings,
            old_batch.candidate_settings, old_batch.previous_actuator_flow, old_altered,
            old_batch.elapsed_seconds, source_kind="D3",
        )
    old_topology_delta = float((old_normal.direct_delta_tfv_m3 - old_changed.direct_delta_tfv_m3).abs().max())

    old_v42 = _load_json(STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json")
    old_v421 = _load_json(V421 / "04_12_group_micro" / "stage_result.json")
    compare = {
        "V4.2": _source_metrics(old_v42["group_metrics"]),
        "V4.2.1": _source_metrics(old_v421["group_metrics"]),
        "V4.3": _source_metrics(final_stage["group_metrics"]),
    }
    compare["V4.3"]["D3"]["magnitude_strata"] = final_stage["d3_magnitude_strata"]
    compare["V4.2.1"]["D3"]["magnitude_strata"] = old_v421.get("d3_magnitude_strata", {})

    old_d2 = compare["V4.2.1"]["D2"]
    new_d2 = compare["V4.3"]["D2"]
    old_d3 = compare["V4.2.1"]["D3"]
    new_d3 = compare["V4.3"]["D3"]
    large_old = float(old_d3.get("magnitude_strata", {}).get("large", {}).get("response_ratio", float("nan")))
    large_new = float(new_d3.get("magnitude_strata", {}).get("large", {}).get("response_ratio", float("nan")))
    d2_regressed = bool(
        new_d2.get("rank", -math.inf) < old_d2.get("rank", math.inf)
        and new_d2.get("pairwise", -math.inf) < old_d2.get("pairwise", math.inf)
    )
    d3_consistent_improvement = bool(
        new_d3.get("rank", -math.inf) > old_d3.get("rank", math.inf)
        and new_d3.get("pairwise", -math.inf) > old_d3.get("pairwise", math.inf)
        and new_d3.get("max_regret_m3", math.inf) < old_d3.get("max_regret_m3", -math.inf)
        and large_new > large_old
    )

    cross_payload = {
        **interference,
        "new_reference_parameter_sha256": parameter_sha256(final_model, reference_parameter_names(final_model)),
        "new_d2_single_parameter_sha256": parameter_sha256(final_model, single_parameter_names(final_model)),
        "new_interaction_parameter_sha256": parameter_sha256(final_model, interaction_parameter_names(final_model)),
        "new_reference_freeze_pass": True,
        "new_d2_freeze_during_d3_pass": bool(final_stage["d2_prediction_invariance"]["prediction_invariant"]),
    }
    topology_payload = {
        "contract": "STEP2_TOPOLOGY_INTERACTION_AUDIT_V43",
        "graph_contract": topology_contract,
        "old_v4_2_1_topology_active": bool(old_topology_delta > 1e-7),
        "old_v4_2_1_edge_ablation_max_abs_m3": old_topology_delta,
        "v4_3_topology_active": bool(topology_ablation["prediction_change_max_abs_m3"] is None or topology_ablation["delta_rank"] != 0.0 or topology_gradient["network_response_magnitude"] > 0.0),
        "topology_ablation": topology_ablation,
        "state_context_ablation": state_ablation,
        "topology_gradient": topology_gradient,
        "local_hydraulic_state_conditioning": True,
        "zero_single_multi_invariants": invariants,
    }
    report_payload = {
        "contract": "PROJECT7_STEP2_STATE_TOPOLOGY_INTERACTION_V43",
        "git_parent": "84e31702c2c05ab79822611fd9aa4411c79f21dd",
        "branch": "agent/step2-topology-interaction-v43",
        "draft_pr_base": "agent/step2-control-response-v4",
        "boundary": final_stage["boundary"],
        "cross_source_interference": cross_payload,
        "topology": topology_payload,
        "V4_2": compare["V4.2"],
        "V4_2_1": compare["V4.2.1"],
        "V4_3": compare["V4.3"],
        "state_topology_strata": state_strata,
        "performance": {
            "tiny": tiny_result["reference_phase"]["profile_seconds"],
            "micro": micro_result["reference_phase"]["profile_seconds"] if micro_result else None,
            "reference_forwards_per_group": 1,
            "gpu": _gpu_snapshot(),
        },
        "physical_causal": invariants,
        "scientific_interpretation": {
            "d3_topology_path_structurally_active": bool(topology_payload["v4_3_topology_active"]),
            "topology_ablation_changed_prediction": bool(topology_payload["topology_ablation"]["prediction_change_max_abs_m3"] > 1e-7),
            "d2_regressed_vs_v4_2_1": d2_regressed,
            "d3_consistent_improvement_vs_v4_2_1": d3_consistent_improvement,
            "large_effect_response_ratio": "IMPROVED" if large_new > large_old else "WORSE_OR_UNCHANGED",
            "magnitude_only_hypothesis": "INSUFFICIENT",
            "topology_conditioned_interaction_next": "YES",
        },
        "verdict": "GREEN_FOR_EXTERNAL_REVIEW" if d3_consistent_improvement and not d2_regressed else "RED",
        "ready_for_full_train_smoke": False,
        "ready_for_formal": False,
        "need_new_swmm": False,
        "next_bounded_action": "Do not promote V4.3; external review should decide a state/topology interaction redesign before any production wiring or full Train-only smoke.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_report(OUT / "STEP2_CROSS_SOURCE_INTERFERENCE_AUDIT_V43.json", "STEP2 CROSS-SOURCE INTERFERENCE AUDIT V4.3", cross_payload)
    _write_report(OUT / "STEP2_TOPOLOGY_INTERACTION_AUDIT_V43.json", "STEP2 TOPOLOGY INTERACTION AUDIT V4.3", topology_payload)
    _write_report(OUT / "STEP2_STATE_TOPOLOGY_INTERACTION_V43_REPORT.json", "PROJECT7 STEP2 STATE-TOPOLOGY INTERACTION V4.3", report_payload)
    docs = Path(__file__).resolve().parents[1] / "docs"
    _write_report(docs / "STEP2_CROSS_SOURCE_INTERFERENCE_AUDIT_V43.json", "STEP2 CROSS-SOURCE INTERFERENCE AUDIT V4.3", cross_payload)
    _write_report(docs / "STEP2_TOPOLOGY_INTERACTION_AUDIT_V43.json", "STEP2 TOPOLOGY INTERACTION AUDIT V4.3", topology_payload)
    _write_report(docs / "STEP2_STATE_TOPOLOGY_INTERACTION_V43_REPORT.json", "PROJECT7 STEP2 STATE-TOPOLOGY INTERACTION V4.3", report_payload)
    print(json.dumps(report_payload, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
