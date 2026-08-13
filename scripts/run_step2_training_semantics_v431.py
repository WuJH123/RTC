"""Run the bounded V4.3.1 training-semantics attribution experiment.

The runner only consumes the existing development/train cache and frozen tiny /
12-group cohorts.  It does not launch SWMM and never reads Validation or Final.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_step2_state_topology_interaction_v43 as legacy  # noqa: E402
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
    stack_response_group_v41,
)
from rtc.step2_train_response_v431 import (  # noqa: E402
    TRAINING_CONTRACT_V431,
    current_state_diagnostics_v431,
    d3_loss_gradient_conflict_audit_v431,
    train_phase_v431,
    validate_fresh_parent_lineage_v431,
)


STUDY = legacy.STUDY
V4 = legacy.V4
V421 = legacy.V421
OUT = STUDY / "step2_training_semantics_v431"
CACHE = legacy.CACHE
GRAPH = legacy.GRAPH
V3_CHECKPOINT = legacy.V3_CHECKPOINT
OLD_SCALE = legacy.OLD_SCALE
SCALE_JSON = legacy.SCALE_JSON
PARENT = legacy.PARENT.resolve()
TINY_GROUPS = legacy.TINY_GROUPS
MICRO_GROUPS = legacy.MICRO_GROUPS
EXPECTED_MICRO_SHA256 = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


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


def _source_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {source: _aggregate([row for row in rows if row["source_kind"] == source]) for source in ("D2", "D3")}


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _all_d2_snapshots(model: torch.nn.Module, groups: dict[str, list[Any]], selected: list[str], prepared: Any, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    model.eval()
    snapshots: dict[str, dict[str, torch.Tensor]] = {}
    for group in selected:
        if not group.startswith("D2::"):
            continue
        batch = stack_response_group_v41(groups[group], device)
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D2",
            )
        snapshots[group] = {
            "delta_states": output.delta_states_physical.detach().clone(),
            "delta_flows": output.delta_flows_physical.detach().clone(),
            "direct_tfv": output.direct_delta_tfv_m3.detach().clone(),
            "trajectory_tfv": output.trajectory_delta_tfv_m3.detach().clone(),
        }
    return snapshots


def _compare_snapshots(before: dict[str, dict[str, torch.Tensor]], after: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    invariant = True
    for group in sorted(before):
        differences = {
            key: float((before[group][key] - after[group][key]).abs().max().item())
            for key in before[group]
        }
        rows[group] = differences
        invariant = invariant and all(value <= 1e-7 for value in differences.values())
    rows["prediction_invariant"] = invariant
    return rows


def _run_experiment(
    *,
    name: str,
    selected: list[str],
    groups: dict[str, list[Any]],
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    topology: bool,
    parent: Path,
    epochs: tuple[int, int, int],
    device_name: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    target_device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    builder = legacy._build_v43 if topology else legacy._build_v41
    model = builder(graph, normalization, scales).to(target_device).float()
    parent_load = legacy._load_compatible(model, parent, allow_topology_missing=topology)
    selected_groups = {group: groups[group] for group in selected}
    d2_groups = {group: groups[group] for group in selected if group.startswith("D2::")}
    d3_groups = {group: groups[group] for group in selected if group.startswith("D3::")}
    reference_groups = legacy.deduplicate_reference_groups_v43(selected_groups)
    phase_root = OUT / name
    phase_root.mkdir(parents=True, exist_ok=True)
    reference_result = train_phase_v431(
        model=model, grouped_pairs=reference_groups, normalization=normalization, scales=scales,
        graph=graph, phase="reference", out_path=phase_root / "reference_phase.pt",
        epochs=epochs[0], learning_rate=2e-3, device=device_name, seed=42,
    )
    d2_result = train_phase_v431(
        model=model, grouped_pairs=d2_groups, normalization=normalization, scales=scales,
        graph=graph, phase="d2", out_path=phase_root / "d2_phase.pt",
        epochs=epochs[1], learning_rate=2e-3, device=device_name, seed=42,
    )
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    d2_before_metrics, _ = evaluate_response_groups_v41(
        model=model, grouped_pairs=d2_groups, prepared=prepared, device=target_device
    )
    d2_before = _all_d2_snapshots(model, groups, selected, prepared, target_device)
    ref_sha_before = parameter_sha256(model, reference_parameter_names(model))
    single_sha_before = parameter_sha256(model, single_parameter_names(model))
    interaction_sha_before = parameter_sha256(model, interaction_parameter_names(model))
    d3_result = train_phase_v431(
        model=model, grouped_pairs=d3_groups, normalization=normalization, scales=scales,
        graph=graph, phase="d3", out_path=phase_root / "d3_phase.pt",
        epochs=epochs[2], learning_rate=2e-3, device=device_name, seed=42,
    )
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    d2_after = _all_d2_snapshots(model, groups, selected, prepared, target_device)
    d2_invariance = _compare_snapshots(d2_before, d2_after)
    ref_sha_after = parameter_sha256(model, reference_parameter_names(model))
    single_sha_after = parameter_sha256(model, single_parameter_names(model))
    interaction_sha_after = parameter_sha256(model, interaction_parameter_names(model))
    metric_rows, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=selected_groups, prepared=prepared, device=target_device
    )
    d3_strata = magnitude_strata_metrics_v41(
        contributions,
        q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]),
        q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]),
    )
    checkpoint = phase_root / f"{name}.pt"
    torch.save({"contract": TRAINING_CONTRACT_V431, "model_state_dict": model.state_dict(), "phase": "complete", "seed": 42, "immutable_parent": str(parent)}, checkpoint)
    result = {
        "contract": TRAINING_CONTRACT_V431,
        "stage": name,
        "architecture": "V4.3 topology" if topology else "V4.2.1 no-topology control",
        "selected_groups": selected,
        "parent_checkpoint": str(parent),
        "parent_load": parent_load,
        "reference_group_count_raw": len(selected_groups),
        "reference_group_count_unique": len(reference_groups),
        "reference_deduplication_ratio": float(len(reference_groups) / max(len(selected_groups), 1)),
        "reference_phase": reference_result.__dict__,
        "d2_phase": d2_result.__dict__,
        "d3_phase": d3_result.__dict__,
        "d2_metrics_before_d3": _source_metrics(d2_before_metrics),
        "group_metrics": metric_rows,
        "candidate_contributions": contributions,
        "d3_magnitude_strata": d3_strata,
        "d2_prediction_before_d3_digest": {group: {key: _tensor_digest(value) for key, value in row.items()} for group, row in d2_before.items()},
        "d2_prediction_after_d3_digest": {group: {key: _tensor_digest(value) for key, value in row.items()} for group, row in d2_after.items()},
        "d2_prediction_invariance": d2_invariance,
        "reference_parameter_sha_before_d3": ref_sha_before,
        "reference_parameter_sha_after_d3": ref_sha_after,
        "single_parameter_sha_before_d3": single_sha_before,
        "single_parameter_sha_after_d3": single_sha_after,
        "interaction_parameter_sha_before_d3": interaction_sha_before,
        "interaction_parameter_sha_after_d3": interaction_sha_after,
        "reference_sha_unchanged": ref_sha_before == ref_sha_after,
        "single_sha_unchanged": single_sha_before == single_sha_after,
        "interaction_sha_changed": interaction_sha_before != interaction_sha_after,
        "checkpoint": str(checkpoint),
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


def _reuse_experiment(
    *, name: str, topology: bool, graph: Any, normalization: Any,
    scales: CounterfactualDeltaScalesV41, device_name: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reload a completed fresh experiment without changing its weights."""

    target_device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    builder = legacy._build_v43 if topology else legacy._build_v41
    model = builder(graph, normalization, scales).to(target_device).float()
    stage_root = OUT / name
    payload = legacy._load_json(stage_root / "stage_result.json")
    legacy._load_compatible(model, Path(payload["checkpoint"]), allow_topology_missing=topology)
    return model, payload


def _four_way_state_ablation(model: Any, grouped_pairs: dict[str, list[Any]], prepared: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    configurations = {
        "A_endpoint_on_message_on": (True, True),
        "B_endpoint_off_message_on": (False, True),
        "C_endpoint_on_message_off": (True, False),
        "D_endpoint_off_message_off": (False, False),
    }
    predictions: dict[str, list[np.ndarray]] = {key: [] for key in configurations}
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in configurations}
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, device)
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        for label, (endpoint, message) in configurations.items():
            model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=endpoint, message_state_enabled=message)
            with torch.no_grad():
                output = model.forward_group(
                    batch.initial_state, batch.rainfall, batch.reference_settings,
                    batch.candidate_settings, batch.previous_actuator_flow, prepared,
                    batch.elapsed_seconds, source_kind="D3",
                )
            prediction = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
            predictions[label].append(prediction)
            rows[label].append(group_metrics_v41(predicted=prediction, truth=truth, group=group, source_kind="D3"))
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=True)
    full = np.concatenate(predictions["A_endpoint_on_message_on"]) if predictions["A_endpoint_on_message_on"] else np.zeros(1)
    result: dict[str, Any] = {}
    for label in configurations:
        values = np.concatenate(predictions[label]) if predictions[label] else np.zeros(1)
        result[label] = {
            "metrics": _aggregate(rows[label]),
            "prediction_max_change_vs_A_m3": float(np.max(np.abs(values - full))) if values.shape == full.shape else float("nan"),
        }
    return result


def _topology_ablation(model: Any, grouped_pairs: dict[str, list[Any]], prepared: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    normal_rows: list[dict[str, Any]] = []
    disabled_rows: list[dict[str, Any]] = []
    max_change = 0.0
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, device)
        model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=True)
        with torch.no_grad():
            normal = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
        model.set_topology_ablation(graph_enabled=False, endpoint_state_enabled=True, message_state_enabled=True)
        with torch.no_grad():
            disabled = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        normal_pred = normal.direct_delta_tfv_m3[0].detach().cpu().numpy()
        disabled_pred = disabled.direct_delta_tfv_m3[0].detach().cpu().numpy()
        max_change = max(max_change, float(np.max(np.abs(normal_pred - disabled_pred))))
        normal_rows.append(group_metrics_v41(predicted=normal_pred, truth=truth, group=group, source_kind="D3"))
        disabled_rows.append(group_metrics_v41(predicted=disabled_pred, truth=truth, group=group, source_kind="D3"))
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=True)
    normal = _aggregate(normal_rows)
    disabled = _aggregate(disabled_rows)
    return {
        "normal": normal,
        "graph_disabled": disabled,
        "prediction_change_max_abs_m3": max_change,
        "delta_rank": normal.get("rank", float("nan")) - disabled.get("rank", float("nan")),
        "delta_pairwise": normal.get("pairwise", float("nan")) - disabled.get("pairwise", float("nan")),
        "delta_mean_regret_m3": normal.get("mean_regret_m3", float("nan")) - disabled.get("mean_regret_m3", float("nan")),
    }


def _current_state_strata(groups: dict[str, list[Any]], normalization: Any, graph: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, pairs in sorted(groups.items()):
        if not group.startswith("D3::"):
            continue
        diagnostic = current_state_diagnostics_v431(pairs[0], normalization, graph)
        diagnostic["group"] = group
        rows.append(diagnostic)
    return rows


def _pooling_audit(model: Any, grouped_pairs: dict[str, list[Any]], prepared: Any, device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    model.eval()
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D3::"):
            continue
        batch = stack_response_group_v41(pairs, device)
        with torch.no_grad():
            output = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
        node_response = output.interaction_delta_states_physical[..., 2].abs()
        mean = float(node_response.mean().item())
        rows.append({"group": group, "mean_abs_node_response": mean, "p90_abs_node_response": float(torch.quantile(node_response.flatten(), 0.90).item()), "max_abs_node_response": float(node_response.max().item()), "max_over_mean": float(node_response.max().item() / max(mean, 1e-9))})
    return {"pooling": "node_mean_in_direct_tfv_head", "groups": rows, "large_effect_high_node_concentration": bool(any(row["max_over_mean"] > 10.0 for row in rows))}


def _edge_feature_audit(graph: Any) -> dict[str, Any]:
    names = [name for name in dir(graph) if not name.startswith("_")]
    candidates = [name for name in names if any(token in name.lower() for token in ("edge", "link", "conduit", "roughness", "diameter", "slope", "length"))]
    features = {name: type(getattr(graph, name)).__name__ for name in candidates if name not in {"edge_index"}}
    static_names = tuple(graph.static_node_feature_names)
    node_conveyance = [name for name in static_names if name.startswith("conduit_")]
    return {
        "edge_features_available": False,
        "edge_features_aligned_to_edge_index": False,
        "candidate_graph_attributes": features,
        "node_conduit_aggregate_features": node_conveyance,
        "status": "EDGE_HYDRAULIC_FEATURES_NOT_IN_CURRENT_GRAPH_CONTRACT",
    }


def _compact_stage(result: dict[str, Any]) -> dict[str, Any]:
    """Keep committed reports small; full candidate rows remain in study artifacts."""

    return {
        "stage": result.get("stage"),
        "architecture": result.get("architecture"),
        "selected_group_count": len(result.get("selected_groups", [])),
        "reference_group_count_raw": result.get("reference_group_count_raw"),
        "reference_group_count_unique": result.get("reference_group_count_unique"),
        "reference_deduplication_ratio": result.get("reference_deduplication_ratio"),
        "d2_metrics_before_d3": result.get("d2_metrics_before_d3"),
        "final_metrics": _source_metrics(result.get("group_metrics", [])),
        "d3_magnitude_strata": result.get("d3_magnitude_strata"),
        "d2_prediction_invariance": result.get("d2_prediction_invariance"),
        "reference_sha_unchanged": result.get("reference_sha_unchanged"),
        "single_sha_unchanged": result.get("single_sha_unchanged"),
        "interaction_sha_changed": result.get("interaction_sha_changed"),
        "reference_forwards_per_group": result.get("reference_forwards_per_group"),
        "profile_seconds": {
            phase: result.get(f"{phase}_phase", {}).get("profile_seconds", {})
            for phase in ("reference", "d2", "d3")
        },
    }


def _write_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    with path.with_suffix(".md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    legacy._validate_train_only_cache()
    tiny = _read_groups(TINY_GROUPS)
    micro = _read_groups(MICRO_GROUPS)
    if len(tiny) != 2 or len(micro) != 12 or not set(tiny).issubset(micro):
        raise RuntimeError("frozen V4 cohort contract changed")
    scales = CounterfactualDeltaScalesV41.from_json_dict(legacy._load_json(SCALE_JSON))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = legacy._load_graph(GRAPH)
    groups = load_train_groups(CACHE, normalization, sorted(set(micro)))
    tiny_sha = _sha256(TINY_GROUPS)
    micro_sha = _sha256(MICRO_GROUPS)
    lineage = {
        "immutable_parent": str(PARENT),
        "tiny_parent": str(PARENT),
        "micro_parent": str(PARENT),
        "same_parent": validate_fresh_parent_lineage_v431(PARENT, PARENT, PARENT),
        "micro_loaded_tiny_checkpoint": False,
        "tiny_groups_sha256": tiny_sha,
        "micro_groups_sha256": micro_sha,
        "micro_groups_sha_matches_prior": micro_sha == EXPECTED_MICRO_SHA256,
    }
    if args.reuse_existing and all((OUT / name / "stage_result.json").is_file() for name in ("tiny_control", "tiny_topology", "micro_control", "micro_topology")):
        tiny_control_model, tiny_control = _reuse_experiment(name="tiny_control", topology=False, graph=graph, normalization=normalization, scales=scales, device_name=args.device)
        tiny_topology_model, tiny_topology = _reuse_experiment(name="tiny_topology", topology=True, graph=graph, normalization=normalization, scales=scales, device_name=args.device)
        control_model, control = _reuse_experiment(name="micro_control", topology=False, graph=graph, normalization=normalization, scales=scales, device_name=args.device)
        topology_model, topology = _reuse_experiment(name="micro_topology", topology=True, graph=graph, normalization=normalization, scales=scales, device_name=args.device)
    else:
        tiny_control_model, tiny_control = _run_experiment(name="tiny_control", selected=tiny, groups=groups, graph=graph, normalization=normalization, scales=scales, topology=False, parent=PARENT, epochs=(3, 8, 15), device_name=args.device)
        tiny_topology_model, tiny_topology = _run_experiment(name="tiny_topology", selected=tiny, groups=groups, graph=graph, normalization=normalization, scales=scales, topology=True, parent=PARENT, epochs=(3, 8, 15), device_name=args.device)
        control_model, control = _run_experiment(name="micro_control", selected=micro, groups=groups, graph=graph, normalization=normalization, scales=scales, topology=False, parent=PARENT, epochs=(2, 5, 12), device_name=args.device)
        topology_model, topology = _run_experiment(name="micro_topology", selected=micro, groups=groups, graph=graph, normalization=normalization, scales=scales, topology=True, parent=PARENT, epochs=(2, 5, 12), device_name=args.device)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    prepared = prepare_graph_v41(topology_model, graph, normalization, device)
    d3_groups = {group: groups[group] for group in micro if group.startswith("D3::")}
    topology_ablation = _topology_ablation(topology_model, d3_groups, prepared, device)
    state_ablation = _four_way_state_ablation(topology_model, d3_groups, prepared, device)
    loss_audit = d3_loss_gradient_conflict_audit_v431(model=topology_model, grouped_pairs=d3_groups, normalization=normalization, scales=scales, graph=graph, device=args.device)
    graph_contract = topology_graph_contract(prepared)
    edge_audit = _edge_feature_audit(graph)
    current_state = _current_state_strata(d3_groups, normalization, graph)
    pooling = _pooling_audit(topology_model, d3_groups, prepared, device)
    d3_tiny_group = next(group for group in tiny if group.startswith("D3::"))
    physical_causal = legacy._physical_causal_invariants(
        topology_model, groups[d3_tiny_group], prepared, device
    )
    # V4.2.1 has no topology path: changing the supplied edge tensor is a read-only control.
    old_model = legacy._build_v41(graph, normalization, scales).to(device).float()
    legacy._load_compatible(old_model, PARENT, allow_topology_missing=False)
    old_prepared = prepare_graph_v41(old_model, graph, normalization, device)
    representative = stack_response_group_v41(next(iter(d3_groups.values())), device)
    with torch.no_grad():
        old_a = old_model.forward_group(representative.initial_state, representative.rainfall, representative.reference_settings, representative.candidate_settings, representative.previous_actuator_flow, old_prepared, representative.elapsed_seconds, source_kind="D3")
        old_alt_prepared = old_model.prepare_static(static_node_features=old_prepared.static_node_features, actuator_physics=old_prepared.actuator_physics, actuator_upstream=old_prepared.actuator_upstream, actuator_downstream=old_prepared.actuator_downstream, edge_index=old_prepared.edge_index[:, ::2], invert_elevation_m=old_prepared.invert_elevation_m)
        old_b = old_model.forward_group(representative.initial_state, representative.rainfall, representative.reference_settings, representative.candidate_settings, representative.previous_actuator_flow, old_alt_prepared, representative.elapsed_seconds, source_kind="D3")
    old_topology_delta = float((old_a.direct_delta_tfv_m3 - old_b.direct_delta_tfv_m3).abs().max().item())
    old_v42 = legacy._load_json(STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json")
    old_v421 = legacy._load_json(V421 / "04_12_group_micro" / "stage_result.json")
    old_v43 = legacy._load_json(STUDY / "step2_state_topology_interaction_v43" / "micro" / "stage_result.json")
    compare = {
        "OLD_V4.2": _source_metrics(old_v42["group_metrics"]),
        "OLD_V4.2.1_CORRECTED": _source_metrics(old_v421["group_metrics"]),
        "OLD_V4.3": _source_metrics(old_v43["group_metrics"]),
        "V4.3.1_PHASED_NO_TOPOLOGY_CONTROL": _source_metrics(control["group_metrics"]),
        "V4.3.1_PHASED_TOPOLOGY": _source_metrics(topology["group_metrics"]),
    }
    compare["OLD_V4.2.1_CORRECTED"]["D3"]["magnitude_strata"] = old_v421.get("d3_magnitude_strata", {})
    compare["OLD_V4.3"]["D3"]["magnitude_strata"] = old_v43.get("d3_magnitude_strata", {})
    compare["V4.3.1_PHASED_NO_TOPOLOGY_CONTROL"]["D3"]["magnitude_strata"] = control["d3_magnitude_strata"]
    compare["V4.3.1_PHASED_TOPOLOGY"]["D3"]["magnitude_strata"] = topology["d3_magnitude_strata"]
    control_d3 = compare["V4.3.1_PHASED_NO_TOPOLOGY_CONTROL"]["D3"]
    topo_d3 = compare["V4.3.1_PHASED_TOPOLOGY"]["D3"]
    topology_supported = bool(topo_d3.get("rank", -math.inf) > control_d3.get("rank", math.inf) and topo_d3.get("pairwise", -math.inf) > control_d3.get("pairwise", math.inf) and topo_d3.get("max_regret_m3", math.inf) < control_d3.get("max_regret_m3", -math.inf))
    report = {
        "contract": "PROJECT7_STEP2_TRAINING_SEMANTICS_V431",
        "git_parent": "d84a2a45279529fc2ba482d2c7e26acf58d3564b",
        "branch": "agent/step2-training-semantics-v431",
        "draft_pr_base": "agent/step2-topology-interaction-v43",
        "boundary": {"swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False, "validation_outcomes_accessed": False, "final_accessed": False, "formal_run": False, "full_train_smoke_run": False},
        "initialization": lineage,
        "phase_loss": {
            "reference_only": True,
            "d2_only": True,
            "d3_interaction_only": True,
            "reference_counterfactual_gradient": "ZERO_BY_CONSTRUCTION",
            "d2_reference_interaction_gradient": "ZERO_BY_PHASE_FREEZE",
            "loss_gradient_audit": loss_audit,
        },
        "state_diagnostic": {"current_state_source": "initial_state", "future_state_used_as_current": False, "rows": current_state},
        "phased_no_topology_control": _compact_stage(control),
        "phased_topology": _compact_stage(topology),
        "tiny_control": _compact_stage(tiny_control),
        "tiny_topology": _compact_stage(tiny_topology),
        "topology": {"graph_contract": graph_contract, "old_v4_2_1_topology_active": old_topology_delta > 1e-7, "old_edge_ablation_max_abs_m3": old_topology_delta, "topology_ablation": topology_ablation, "state_ablation": state_ablation, "pooling_audit": pooling},
        "physical_causal": physical_causal,
        "edge_hydraulic_audit": edge_audit,
        "compare": compare,
        "topology_net_contribution": {"delta_rank": topo_d3.get("rank", float("nan")) - control_d3.get("rank", float("nan")), "delta_pairwise": topo_d3.get("pairwise", float("nan")) - control_d3.get("pairwise", float("nan")), "delta_mean_regret_m3": topo_d3.get("mean_regret_m3", float("nan")) - control_d3.get("mean_regret_m3", float("nan")), "supported": topology_supported},
        "verdict": "GREEN_FOR_EXTERNAL_REVIEW" if topology_supported else "RED",
        "ready_for_full_train_smoke": False,
        "ready_for_formal": False,
        "need_new_swmm": False,
        "next_bounded_action": "Keep production wiring frozen; if phased control recovers but topology is neutral/harmful, design V4.4 edge-hydraulic-conditioned interaction.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_report(OUT / "STEP2_PHASE_LOSS_AUDIT_V431.json", "STEP2 PHASE LOSS AUDIT V4.3.1", {"phase_loss": report["phase_loss"], "initialization": lineage})
    _write_report(OUT / "STEP2_STATE_ABLATION_AUDIT_V431.json", "STEP2 STATE ABLATION AUDIT V4.3.1", {"state_diagnostic": report["state_diagnostic"], "topology": report["topology"]})
    _write_report(OUT / "STEP2_TRAINING_SEMANTICS_V431_REPORT.json", "PROJECT7 STEP2 TRAINING SEMANTICS V4.3.1", report)
    docs = Path(__file__).resolve().parents[1] / "docs"
    _write_report(docs / "STEP2_PHASE_LOSS_AUDIT_V431.json", "STEP2 PHASE LOSS AUDIT V4.3.1", {"phase_loss": report["phase_loss"], "initialization": lineage})
    _write_report(docs / "STEP2_STATE_ABLATION_AUDIT_V431.json", "STEP2 STATE ABLATION AUDIT V4.3.1", {"state_diagnostic": report["state_diagnostic"], "topology": report["topology"]})
    _write_report(docs / "STEP2_TRAINING_SEMANTICS_V431_REPORT.json", "PROJECT7 STEP2 TRAINING SEMANTICS V4.3.1", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
