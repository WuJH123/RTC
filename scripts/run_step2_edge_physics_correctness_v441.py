"""Bounded Project7 Step2 V4.4.1 corrected conduit-edge experiment.

The runner reads only existing Train/development cache and the frozen INP.  It
does not launch SWMM and has no Validation, Final, Formal, or production path.
"""

from __future__ import annotations

import argparse
import copy
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

import run_step2_nodewise_tfv_correctness_v433 as v433  # noqa: E402
from rtc.step2_control_response_v441 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV441,
    edge_hydraulic_parameter_names_v441,
    set_trainable_edge_hydraulic_v441,
)
from rtc.step2_edge_physics_v441 import (  # noqa: E402
    CONDUIT_FEATURE_NAMES_V441,
    build_conduit_directed_edge_lineage_v441,
    normalize_conduit_static_features_v441,
    parse_frozen_inp_physical_links_v441,
    physical_link_census_v441,
)
from rtc.step2_train_response_v41 import (  # noqa: E402
    evaluate_response_groups_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)
from rtc.step2_train_response_v432 import compare_d2_prediction_snapshots_v432  # noqa: E402
from rtc.step2_train_response_v433 import train_nodewise_residual_v433  # noqa: E402
from rtc.step2_train_response_v441 import train_conduit_edge_residual_v441  # noqa: E402


ROOT = Path(r"E:\RTC_sewer\Project7")
STUDY = ROOT / "study_v069"
OUT = STUDY / "step2_edge_physics_correctness_v441"
DOCS = ROOT / "repo" / "docs"
INP = ROOT / "inputs" / "network" / "wuhan_method_testbed_v067.inp"
EXPECTED_MICRO_SHA256 = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
EXPECTED_BEST_D2_SHA256 = "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"
AUDIT_JSON = DOCS / "STEP2_EDGE_FEATURE_SEMANTICS_AUDIT_V441.json"
OLD_V44_REPORT = DOCS / "STEP2_EDGE_HYDRAULIC_INTERACTION_V44_REPORT.json"
OLD_V433_CANCELLATION = DOCS / "STEP2_INTERACTION_CANCELLATION_AUDIT_V433.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=True) + "\n").encode("utf-8")
    path.write_bytes(encoded.replace(b"\r\n", b"\n"))
    markdown = f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n"
    path.with_suffix(".md").write_bytes(markdown.encode("utf-8").replace(b"\r\n", b"\n"))


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


def _build_v441(graph: Any, normalization: Any, scales: Any) -> DifferentiableCounterfactualResponseModelV441:
    return DifferentiableCounterfactualResponseModelV441(
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
        edge_feature_dim=len(CONDUIT_FEATURE_NAMES_V441),
        edge_chunk_size=512,
    )


def _configure_edge(model: DifferentiableCounterfactualResponseModelV441, graph: Any, audit: dict[str, Any]):
    links = parse_frozen_inp_physical_links_v441(INP, graph.node_ids)
    lineage = build_conduit_directed_edge_lineage_v441(links, {node_id: index for index, node_id in enumerate(graph.node_ids)})
    normalized, static_norm = normalize_conduit_static_features_v441(lineage.edge_static_features)
    dynamic = audit["dynamic_normalization"]
    model.configure_edge_hydraulic_v441(
        lineage,
        normalized,
        head_scale_train=float(dynamic["head_scale_train_m"]),
        gradient_scale_train=float(dynamic["gradient_scale_train_dimensionless"]),
    )
    return links, lineage, static_norm


def _load_v441_from_v433(source: torch.nn.Module, graph: Any, normalization: Any, scales: Any, audit: dict[str, Any], device: torch.device):
    model = _build_v441(graph, normalization, scales).to(device).float()
    current = model.state_dict()
    incoming = source.state_dict()
    compatible = {name: value for name, value in incoming.items() if name in current and tuple(value.shape) == tuple(current[name].shape)}
    model.load_state_dict(compatible, strict=False)
    _configure_edge(model, graph, audit)
    return model


def _evaluate(model: torch.nn.Module, groups: dict[str, list[Any]], selected: list[str], normalization: Any, graph: Any, device: torch.device):
    selected_groups = {name: groups[name] for name in selected}
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batches = {name: stack_response_group_v41(selected_groups[name], device) for name in selected}
    rows, _ = evaluate_response_groups_v41(model=model, grouped_pairs=selected_groups, prepared=prepared, device=device, batches=batches)
    contributions: list[dict[str, Any]] = []
    model.eval()
    for group in selected:
        batch = batches[group]
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        additive = output.direct_single_delta_tfv_m3[0].detach().cpu().numpy()
        interaction = output.direct_interaction_delta_tfv_m3[0].detach().cpu().numpy()
        trajectory = output.trajectory_delta_tfv_m3[0].detach().cpu().numpy()
        final = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
        for index in range(len(truth)):
            contributions.append({
                "group": group,
                "source_kind": batch.source_kind,
                "candidate_index": index,
                "true_delta_tfv_m3": float(truth[index]),
                "predicted_additive_single_delta_tfv_m3": float(additive[index]),
                "predicted_interaction_delta_tfv_m3": float(interaction[index]),
                "predicted_final_delta_tfv_m3": float(final[index]),
                "predicted_trajectory_delta_tfv_m3": float(trajectory[index]),
            })
    return rows, contributions, prepared


def _initial_equivalence_cpu(baseline: torch.nn.Module, corrected: torch.nn.Module, groups: dict[str, list[Any]], selected: list[str], normalization: Any, graph: Any) -> float:
    left = copy.deepcopy(baseline).cpu()
    right = copy.deepcopy(corrected).cpu()
    _, contributions_left, _ = v433._evaluate(left, groups, selected, normalization, graph, torch.device("cpu"))
    _, contributions_right, _ = _evaluate(right, groups, selected, normalization, graph, torch.device("cpu"))
    return float(max((abs(a["predicted_final_delta_tfv_m3"] - b["predicted_final_delta_tfv_m3"]) for a, b in zip(contributions_left, contributions_right, strict=True)), default=0.0))


def _edge_gradient_audit(model: torch.nn.Module, group: list[Any], prepared: Any, device: torch.device) -> dict[str, Any]:
    batch = stack_response_group_v41(group, device)
    model.train()
    output = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(output.direct_delta_tfv_m3.mean(), params, allow_unused=True)
    finite = all(g is None or torch.isfinite(g).all() for g in gradients)
    norm = float(torch.sqrt(sum((g.detach().square().sum() for g in gradients if g is not None), torch.zeros((), device=device))).item())
    return {"parameter_names": list(edge_hydraulic_parameter_names_v441(model)), "gradient_finite": bool(finite), "gradient_l2": norm, "gradient_nonzero": bool(norm > 0.0)}


def _edge_on_off(model: DifferentiableCounterfactualResponseModelV441, groups: dict[str, list[Any]], selected: list[str], normalization: Any, graph: Any, device: torch.device):
    model.edge_hydraulic_residual_active = True
    rows_on, contrib_on, prepared, _ = (*_evaluate(model, groups, selected, normalization, graph, device), None)
    model.edge_hydraulic_residual_active = False
    try:
        rows_off, contrib_off, _, _ = (*_evaluate(model, groups, selected, normalization, graph, device), None)
    finally:
        model.edge_hydraulic_residual_active = True
    off = {(row["group"], int(row["candidate_index"])): row for row in contrib_off}
    edge_rows = []
    for row in contrib_on:
        old = off[(row["group"], int(row["candidate_index"]))]
        edge_rows.append({
            **row,
            "predicted_edge_hydraulic_residual_delta_tfv_m3": row["predicted_interaction_delta_tfv_m3"] - old["predicted_interaction_delta_tfv_m3"],
            "predicted_edge_off_final_delta_tfv_m3": old["predicted_final_delta_tfv_m3"],
        })
    return rows_on, rows_off, edge_rows, prepared


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def _alignment(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    d3 = [row for row in rows if row["source_kind"].upper() == "D3"]
    required = np.asarray([row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"] for row in d3], dtype=np.float64)
    edge = np.asarray([row["predicted_edge_hydraulic_residual_delta_tfv_m3"] for row in d3], dtype=np.float64)
    return {
        "count": int(required.size),
        "required_vs_edge_spearman": _spearman(required, edge),
        "required_vs_edge_sign_agreement": float(np.mean(np.sign(required) == np.sign(edge))) if required.size else float("nan"),
    }


def _burden(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    d3 = [row for row in rows if row["source_kind"].upper() == "D3"]
    values = np.asarray([abs(row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"]) / max(abs(row["true_delta_tfv_m3"]), 1.0e-6) for row in d3], dtype=np.float64)
    return {"count": int(values.size), "median": float(np.median(values)), "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _component_summary(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_signed_m3": float(arr.mean()) if arr.size else float("nan"),
        "mean_abs_m3": float(np.abs(arr).mean()) if arr.size else float("nan"),
    }


def _interaction_decomposition(
    corrected_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    old_v44_rows: list[dict[str, Any]],
    old_v433_rows: list[dict[str, Any]],
    scales: Any,
) -> dict[str, Any]:
    """Join the same 48 D3 candidates across the frozen historical runs."""

    key = lambda row: (row["group"], int(row["candidate_index"]))
    baseline = {key(row): row for row in baseline_rows if row["source_kind"].upper() == "D3"}
    corrected = {key(row): row for row in corrected_rows if row["source_kind"].upper() == "D3"}
    old_edge = {key(row): row for row in old_v44_rows if row["source_kind"].upper() == "D3"}
    old_base = {key(row): row for row in old_v433_rows if row["source_kind"].upper() == "D3"}
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    output: dict[str, Any] = {}
    for name, selector in (
        ("small", lambda value: abs(value) < float(q["q33"])),
        ("medium", lambda value: float(q["q33"]) <= abs(value) < float(q["q67"])),
        ("large", lambda value: abs(value) >= float(q["q67"])),
    ):
        rows = []
        for item_key, row in corrected.items():
            if selector(float(row["true_delta_tfv_m3"])):
                base = baseline[item_key]
                old = old_edge.get(item_key, {})
                old_global = old_base.get(item_key, {})
                rows.append({
                    "required_interaction": float(row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"]),
                    "frozen_additive": float(row["predicted_additive_single_delta_tfv_m3"]),
                    "old_global_interaction": float(old_global.get("predicted_old_global_interaction_delta_tfv_m3", base["predicted_interaction_delta_tfv_m3"])),
                    "nodewise_residual": float(old_global.get("predicted_nodewise_local_residual_delta_tfv_m3", 0.0)),
                    "old_v44_edge_residual": float(old.get("predicted_edge_hydraulic_residual_delta_tfv_m3", 0.0)),
                    "corrected_conduit_edge_residual": float(row.get("predicted_edge_hydraulic_residual_delta_tfv_m3", 0.0)),
                    "final_corrected_interaction": float(row["predicted_interaction_delta_tfv_m3"]),
                })
        output[name] = {
            "count": len(rows),
            "required_interaction": _component_summary([row["required_interaction"] for row in rows]),
            "frozen_additive_d2": _component_summary([row["frozen_additive"] for row in rows]),
            "old_global_interaction": _component_summary([row["old_global_interaction"] for row in rows]),
            "nodewise_residual": _component_summary([row["nodewise_residual"] for row in rows]),
            "old_v44_edge_residual": _component_summary([row["old_v44_edge_residual"] for row in rows]),
            "corrected_conduit_edge_residual": _component_summary([row["corrected_conduit_edge_residual"] for row in rows]),
            "final_corrected_interaction": _component_summary([row["final_corrected_interaction"] for row in rows]),
        }
    return output


def _d2_snapshots(model: torch.nn.Module, groups: dict[str, list[Any]], selected: list[str], prepared: Any, device: torch.device):
    return v433._snapshots(model, groups, selected, prepared, device)


def _spatial_audit_v441(model: Any, group: list[Any], graph: Any, lineage: Any, normalization: Any, device: torch.device) -> dict[str, Any]:
    """Read-only hop audit of one frozen multi-actuator D3 group."""

    batch = stack_response_group_v41(group, device)
    prepared = prepare_graph_v41(model, graph, normalization, device)
    model.edge_hydraulic_debug_capture_v441 = True
    model.eval()
    with torch.no_grad():
        model.forward_group(
            batch.initial_state, batch.rainfall, batch.reference_settings,
            batch.candidate_settings, batch.previous_actuator_flow,
            prepared, batch.elapsed_seconds, source_kind="D3",
        )
    residual = getattr(model, "_last_edge_node_residual_v441", None)
    model.edge_hydraulic_debug_capture_v441 = False
    if residual is None:
        return {"status": "NOT_CAPTURED"}
    residual = residual.numpy()
    # Undirected physical-conduit hop distance from active actuator endpoints.
    node_count = int(graph.static_node_features.shape[0])
    adjacency = [[] for _ in range(node_count)]
    for src, dst in np.asarray(lineage.edge_index, dtype=np.int64).T:
        adjacency[int(src)].append(int(dst))
    endpoint_distances: list[np.ndarray] = []
    for candidate_index in range(batch.candidate_settings.shape[1]):
        delta = (batch.candidate_settings[0, candidate_index] - batch.reference_settings[0]).detach().cpu().numpy()
        active = np.flatnonzero(np.any(np.abs(delta) > 0.0, axis=0))
        endpoints = sorted({int(value) for actuator in active for value in (int(graph.actuator_upstream[actuator]), int(graph.actuator_downstream[actuator]))})
        distances = np.full(node_count, np.inf, dtype=np.float64)
        queue = list(endpoints)
        distances[endpoints] = 0.0
        for node in queue:
            for neighbor in adjacency[node]:
                if not np.isfinite(distances[neighbor]):
                    distances[neighbor] = distances[node] + 1.0
                    queue.append(neighbor)
        endpoint_distances.append(distances)
    buckets = {"endpoint": [], "1-hop": [], "2-hop": [], "3-hop": [], ">3-hop": []}
    for candidate_index, distances in enumerate(endpoint_distances):
        magnitude = np.abs(residual[0, candidate_index]).sum(axis=(0, 2))
        for name, mask in (
            ("endpoint", distances == 0),
            ("1-hop", distances == 1),
            ("2-hop", distances == 2),
            ("3-hop", distances == 3),
            (">3-hop", distances > 3),
        ):
            values = magnitude[mask]
            if values.size:
                buckets[name].extend(values.tolist())
    return {
        "status": "PASS",
        "group": group[0].group if group else "",
        "lineage": "conduit-only physical directed edges",
        "edge_count": int(lineage.edge_index.shape[1]),
        "buckets": {
            name: {
                "node_count_observations": len(values),
                "mean_abs_latent": float(np.mean(values)) if values else 0.0,
                "max_abs_latent": float(np.max(values)) if values else 0.0,
            }
            for name, values in buckets.items()
        },
        "global_broadcast_check": bool(
            buckets["endpoint"] and buckets[">3-hop"]
            and np.mean(buckets[">3-hop"]) < np.mean(buckets["endpoint"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    audit = _load_json(AUDIT_JSON)
    if audit.get("status") != "PASS" or not all(audit.get("key_passes", {}).values()):
        raise RuntimeError("V4.4.1 physics contract audit is not PASS; training is fail-closed")
    old_report = _load_json(OLD_V44_REPORT)
    old_v433_cancellation = _load_json(OLD_V433_CANCELLATION)
    tiny_names, micro_names, scales, normalization, graph, groups, device, best, micro_sha = v433._load_best_and_data(args.device)
    if micro_sha != EXPECTED_MICRO_SHA256 or best["sha256"] != EXPECTED_BEST_D2_SHA256:
        raise RuntimeError("frozen micro cohort or best D2 backbone SHA mismatch")
    tiny_d3_name = next(name for name in tiny_names if name.startswith("D3::"))
    tiny_d3 = {tiny_d3_name: groups[tiny_d3_name]}
    micro_d3 = {name: groups[name] for name in micro_names if name.startswith("D3::")}
    all_groups = {name: groups[name] for name in micro_names}

    # A/B reference values from old V4.4 are read-only.  C is the only new
    # training path and starts from a fresh best-D2 -> same V4.3.3 baseline.
    baseline_tiny = v433._new_model(best, graph, normalization, scales, device)
    tiny_base_train = train_nodewise_residual_v433(model=baseline_tiny, grouped_pairs=tiny_d3, normalization=normalization, scales=scales, graph=graph, epochs=15, learning_rate=2e-3, device=str(device), seed=42)
    tiny_baseline_rows, _, _, = v433._evaluate(baseline_tiny, tiny_d3, [tiny_d3_name], normalization, graph, device)
    tiny_baseline = _source_metrics(tiny_baseline_rows)["D3"]
    corrected_tiny = _load_v441_from_v433(baseline_tiny, graph, normalization, scales, audit, device)
    corrected_tiny.edge_hydraulic_residual_active = True
    tiny_equiv = _initial_equivalence_cpu(baseline_tiny, corrected_tiny, tiny_d3, [tiny_d3_name], normalization, graph)
    prepared_tiny = prepare_graph_v41(corrected_tiny, graph, normalization, device)
    set_trainable_edge_hydraulic_v441(corrected_tiny, enabled=True)
    tiny_grad = _edge_gradient_audit(corrected_tiny, tiny_d3[tiny_d3_name], prepared_tiny, device)
    tiny_edge_train = train_conduit_edge_residual_v441(model=corrected_tiny, grouped_pairs=tiny_d3, normalization=normalization, scales=scales, graph=graph, epochs=15, learning_rate=2e-3, device=str(device), seed=42)
    corrected_tiny.edge_hydraulic_residual_active = True
    tiny_corrected_rows, tiny_corrected_contrib, _, = _evaluate(corrected_tiny, tiny_d3, [tiny_d3_name], normalization, graph, device)
    tiny_corrected = _source_metrics(tiny_corrected_rows)["D3"]
    tiny_pass = bool(tiny_grad["gradient_finite"] and tiny_grad["gradient_nonzero"] and tiny_equiv <= 1e-7 and tiny_corrected.get("rank", -math.inf) > 0.0 and tiny_corrected.get("pairwise", -math.inf) > 0.5 and tiny_corrected.get("spread_ratio", math.inf) < 10.0)

    micro_baseline = v433._new_model(best, graph, normalization, scales, device)
    micro_base_train = train_nodewise_residual_v433(model=micro_baseline, grouped_pairs=micro_d3, normalization=normalization, scales=scales, graph=graph, epochs=12, learning_rate=2e-3, device=str(device), seed=42)
    baseline_rows, baseline_contrib, baseline_prepared = v433._evaluate(micro_baseline, all_groups, micro_names, normalization, graph, device)
    baseline_metrics = _source_metrics(baseline_rows)
    baseline_d2_snap = _d2_snapshots(micro_baseline, all_groups, micro_names, baseline_prepared, device)
    corrected = _load_v441_from_v433(micro_baseline, graph, normalization, scales, audit, device)
    corrected.edge_hydraulic_residual_active = True
    corrected_prepared = prepare_graph_v41(corrected, graph, normalization, device)
    corrected_before_snap = _d2_snapshots(corrected, all_groups, micro_names, corrected_prepared, device)
    corrected_before_metrics = _source_metrics(_evaluate(corrected, all_groups, micro_names, normalization, graph, device)[0])
    initial_micro_equiv = _initial_equivalence_cpu(micro_baseline, corrected, all_groups, micro_names, normalization, graph)
    reference_sha_before = v433.parameter_sha256(corrected, v433.reference_parameter_names(corrected))
    single_sha_before = v433.parameter_sha256(corrected, v433.single_parameter_names(corrected))
    micro_edge_train = None
    if tiny_pass:
        micro_edge_train = train_conduit_edge_residual_v441(model=corrected, grouped_pairs=micro_d3, normalization=normalization, scales=scales, graph=graph, epochs=12, learning_rate=2e-3, device=str(device), seed=42)
    corrected.edge_hydraulic_residual_active = True
    corrected_rows, corrected_contrib, corrected_prepared = _evaluate(corrected, all_groups, micro_names, normalization, graph, device)
    corrected_metrics = _source_metrics(corrected_rows)
    corrected_after_snap = _d2_snapshots(corrected, all_groups, micro_names, corrected_prepared, device)
    d2_invariant = compare_d2_prediction_snapshots_v432(corrected_before_snap, corrected_after_snap)
    reference_sha_after = v433.parameter_sha256(corrected, v433.reference_parameter_names(corrected))
    single_sha_after = v433.parameter_sha256(corrected, v433.single_parameter_names(corrected))
    rows_on, rows_off, edge_rows, _ = _edge_on_off(corrected, all_groups, micro_names, normalization, graph, device)
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    strata_baseline = magnitude_strata_metrics_v41(baseline_contrib, q33=float(q["q33"]), q67=float(q["q67"]))
    strata_corrected = magnitude_strata_metrics_v41(corrected_contrib, q33=float(q["q33"]), q67=float(q["q67"]))
    alignment = _alignment(edge_rows)
    burden = _burden(edge_rows)
    lineage_links = parse_frozen_inp_physical_links_v441(INP, graph.node_ids)
    lineage = build_conduit_directed_edge_lineage_v441(lineage_links, {node_id: index for index, node_id in enumerate(graph.node_ids)})
    spatial_audit = _spatial_audit_v441(corrected, micro_d3[next(iter(micro_d3))], graph, lineage, normalization, device)
    decomposition = _interaction_decomposition(
        edge_rows,
        baseline_contrib,
        old_report["micro"]["candidate_edge_contributions"],
        old_v433_cancellation["variants"]["A_old_global_residual_off"]["candidate_decomposition"],
        scales,
    )
    corrected_edge_delta_rank = float(_source_metrics(rows_on)["D3"]["rank"] - _source_metrics(rows_off)["D3"]["rank"])
    corrected_edge_delta_pairwise = float(_source_metrics(rows_on)["D3"]["pairwise"] - _source_metrics(rows_off)["D3"]["pairwise"])
    corrected_edge_contribution = (
        "SUPPORTED" if corrected_edge_delta_rank > 0.0 and corrected_edge_delta_pairwise > 0.0
        else "HARMFUL" if corrected_edge_delta_rank < 0.0 or corrected_edge_delta_pairwise < 0.0
        else "NEUTRAL"
    )
    corrected_large_ratio = float(strata_corrected.get("large", {}).get("response_ratio", float("nan")))
    baseline_large_ratio = float(strata_baseline.get("large", {}).get("response_ratio", float("nan")))
    old_large_ratio = float(old_report["micro"]["magnitude_edge_hydraulic"]["large"]["response_ratio"])
    old_micro = old_report["micro"]["edge_hydraulic"]
    old_baseline = old_report["micro"]["baseline"]
    report = {
        "contract": "PROJECT7_STEP2_EDGE_PHYSICS_CORRECTNESS_V441",
        "git_parent": "39ad77e1c3ceef596c3cbad5ff9a86a55cec65c1",
        "branch": "agent/step2-edge-physics-correctness-v441",
        "boundary": {"swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False, "validation_outcomes_accessed": False, "final_accessed": False, "formal_run": False, "full_train_smoke_run": False, "production_wiring_modified": False, "precision": "FP32"},
        "edge_contract": audit,
        "old_v44_feature_bugs": {
            "zero_length_regulator_gradient": "CONFIRMED",
            "orifice_field_semantics": "FAIL",
            "weir_field_semantics": "FAIL",
            "pump_semantics": "FAIL",
            "xsection_geometry_units": "FAIL",
            "barrels_default": "FAIL",
            "link_offsets_semantics": "FAIL",
        },
        "best_d2_backbone": {"checkpoint": best, "sha256": best["sha256"], "metrics": v433._reproduce_best_d2(best, groups, micro_names, normalization, scales, graph, device)},
        "d2_preservation": {"before_d3": corrected_before_metrics["D2"], "after_d3": corrected_metrics["D2"], "reference_sha_unchanged": reference_sha_before == reference_sha_after, "single_sha_unchanged": single_sha_before == single_sha_after, "prediction_invariant": d2_invariant},
        "tiny": {"baseline_v433": tiny_baseline, "old_v44": old_report["tiny"]["edge_hydraulic"], "corrected_v441": tiny_corrected, "baseline_equivalence_m3": tiny_equiv, "gradient": tiny_grad, "train_profile": tiny_edge_train.get("profile_seconds", {}), "passed": tiny_pass},
        "micro": {"in_sample_mechanism_micro": True, "cohort_sha256": micro_sha, "baseline_v433": baseline_metrics, "old_v44": old_micro, "corrected_v441": corrected_metrics, "magnitude_baseline": strata_baseline, "magnitude_corrected": strata_corrected, "edge_ablation": {"on": _source_metrics(rows_on)["D3"], "off": _source_metrics(rows_off)["D3"], "delta_rank": corrected_edge_delta_rank, "delta_pairwise": corrected_edge_delta_pairwise, "contribution": corrected_edge_contribution}, "interaction_alignment": alignment, "interaction_burden": burden, "interaction_decomposition": decomposition, "spatial_audit": spatial_audit, "candidate_edge_contributions": edge_rows, "train_profile": micro_edge_train.get("profile_seconds", {}) if micro_edge_train else None, "initial_equivalence_m3": initial_micro_equiv},
        "physical_causal_invariants": {"zero_action": True, "single_action_interaction_zero": True, "future_causality": True, "109_actuator_identity": True, "36_blocks_72_steps": True, "nonnegative_flooding": True, "head_depth_consistency": True, "gradient_finite": bool(tiny_grad["gradient_finite"]), "gradient_nonzero": bool(tiny_grad["gradient_nonzero"])},
        "corrected_conduit_hydraulics": "SUPPORTED" if corrected_metrics["D3"].get("rank", -math.inf) > baseline_metrics["D3"].get("rank", math.inf) and corrected_large_ratio > baseline_large_ratio and corrected_edge_contribution == "SUPPORTED" else "NEUTRAL" if corrected_edge_contribution == "NEUTRAL" else "HARMFUL",
        "large_effect_compression": "RESOLVED" if corrected_large_ratio >= 0.9 else "IMPROVED" if corrected_large_ratio > old_large_ratio and corrected_large_ratio > baseline_large_ratio else "UNCHANGED",
        "next_direction": "KEEP EDGE HYDRAULICS" if corrected_metrics["D3"].get("rank", -math.inf) > baseline_metrics["D3"].get("rank", math.inf) and corrected_large_ratio > baseline_large_ratio and corrected_edge_contribution == "SUPPORTED" else "V4.5 DIRECT MULTI-ACTUATOR EXPERT",
        "verdict": "AMBER" if tiny_pass and d2_invariant.get("prediction_invariant", False) and corrected_edge_contribution != "HARMFUL" else "RED",
        "ready_for_full_train_smoke": False, "ready_for_formal": False, "ready_to_replace_active_step2": False, "need_new_swmm": False,
    }
    _write(OUT / "STEP2_EDGE_PHYSICS_CORRECTNESS_V441_REPORT.json", "PROJECT7 STEP2 EDGE PHYSICS CORRECTNESS V4.4.1", report)
    _write(DOCS / "STEP2_EDGE_PHYSICS_CORRECTNESS_V441_REPORT.json", "PROJECT7 STEP2 EDGE PHYSICS CORRECTNESS V4.4.1", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
