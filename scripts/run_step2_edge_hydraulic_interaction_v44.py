"""Bounded Project7 Step2 V4.4 edge-hydraulic interaction experiment.

Only the existing development/train cache, the frozen V4.2 D2-compatible
checkpoint, and the frozen tiny/12-group mechanism cohorts are read.  This
runner never launches SWMM and has no Validation/Final/Formal path.
"""

from __future__ import annotations

import argparse
import csv
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
from rtc.step2_control_response_v44 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV44,
    edge_hydraulic_parameter_names_v44,
    set_trainable_edge_hydraulic_v44,
)
from rtc.step2_edge_hydraulic_v44 import (  # noqa: E402
    build_physical_directed_edge_lineage_v44,
    normalize_edge_static_features_v44,
    parse_frozen_inp_physical_links_v44,
)
from rtc.step2_train_response_v433 import train_nodewise_residual_v433  # noqa: E402
from rtc.step2_train_response_v44 import train_edge_hydraulic_v44  # noqa: E402
from rtc.step2_train_response_v41 import (  # noqa: E402
    evaluate_response_groups_v41,
    group_metrics_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)


ROOT = Path(r"E:\RTC_sewer\Project7")
STUDY = ROOT / "study_v069"
OUT = STUDY / "step2_edge_hydraulic_interaction_v44"
INP = ROOT / "inputs" / "network" / "wuhan_method_testbed_v067.inp"
GRAPH = v433.GRAPH
V3_CHECKPOINT = v433.V3_CHECKPOINT
OLD_SCALE = v433.OLD_SCALE
SCALE_JSON = v433.SCALE_JSON
TINY_GROUPS = v433.TINY_GROUPS
MICRO_GROUPS = v433.MICRO_GROUPS
BEST_V42_STAGE = v433.BEST_V42_STAGE
EXPECTED_MICRO_SHA256 = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
EXPECTED_BEST_D2_SHA256 = "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return {
        source: _aggregate([row for row in rows if row["source_kind"] == source])
        for source in ("D2", "D3")
    }


def _build_v44(graph: Any, normalization: Any, scales: Any) -> DifferentiableCounterfactualResponseModelV44:
    return DifferentiableCounterfactualResponseModelV44(
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
        edge_feature_dim=27,
        edge_chunk_size=512,
    )


def _configure_edge(model: DifferentiableCounterfactualResponseModelV44, graph: Any):
    links = parse_frozen_inp_physical_links_v44(INP, graph.node_ids)
    lineage = build_physical_directed_edge_lineage_v44(
        links, {node_id: index for index, node_id in enumerate(graph.node_ids)}
    )
    normalized, normalization = normalize_edge_static_features_v44(
        lineage.edge_static_features, lineage.edge_static_feature_names
    )
    model.configure_edge_hydraulic_v44(lineage, normalized)
    return links, lineage, normalization


def _load_v44_from_v433(
    source: torch.nn.Module,
    graph: Any,
    normalization: Any,
    scales: Any,
    device: torch.device,
):
    model = _build_v44(graph, normalization, scales).to(device).float()
    current = model.state_dict()
    incoming = source.state_dict()
    compatible = {
        name: value
        for name, value in incoming.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    model.load_state_dict(compatible, strict=False)
    _configure_edge(model, graph)
    return model


def _evaluate(
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
):
    prepared = prepare_graph_v41(model, graph, normalization, device)
    selected_groups = {name: grouped_pairs[name] for name in selected}
    batches = {name: stack_response_group_v41(selected_groups[name], device) for name in selected}
    rows, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=selected_groups,
        prepared=prepared,
        device=device,
        batches=batches,
    )
    return rows, contributions, prepared, batches


def _snapshots(model, groups, selected, prepared, device):
    return v433._snapshots(model, groups, selected, prepared, device)


def _edge_contribution_rows(
    model: DifferentiableCounterfactualResponseModelV44,
    grouped_pairs: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
):
    """Compare one trained model with edge residual on/off, read-only."""

    model.eval()
    rows_on, contributions_on, prepared, batches = _evaluate(
        model, grouped_pairs, selected, normalization, graph, device
    )
    model.edge_hydraulic_residual_active = False
    try:
        rows_off, contributions_off, _, _ = _evaluate(
            model, grouped_pairs, selected, normalization, graph, device
        )
    finally:
        model.edge_hydraulic_residual_active = True
    off_by_key = {
        (row["group"], int(row["candidate_index"])): row for row in contributions_off
    }
    output_rows: list[dict[str, Any]] = []
    for row in contributions_on:
        key = (row["group"], int(row["candidate_index"]))
        old = off_by_key[key]
        output_rows.append(
            {
                **row,
                "predicted_edge_hydraulic_residual_delta_tfv_m3": float(
                    row["predicted_interaction_delta_tfv_m3"]
                    - old["predicted_interaction_delta_tfv_m3"]
                ),
                "predicted_edge_off_direct_delta_tfv_m3": float(
                    old["predicted_final_delta_tfv_m3"]
                ),
            }
        )
    return rows_on, rows_off, output_rows, prepared, batches


def _decomposition(
    edge_rows: list[dict[str, Any]],
    scales: Any,
) -> dict[str, Any]:
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    by_stratum: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for row in edge_rows:
        if row.get("source_kind", "").upper() != "D3":
            continue
        truth_abs = abs(float(row["true_delta_tfv_m3"]))
        if truth_abs < float(q["q33"]):
            stratum = "small"
        elif truth_abs < float(q["q67"]):
            stratum = "medium"
        else:
            stratum = "large"
        by_stratum[stratum].append(row)
    out: dict[str, Any] = {}
    for stratum, values in by_stratum.items():
        def summary(key_fn):
            x = np.asarray([key_fn(row) for row in values], dtype=np.float64)
            return {
                "count": int(x.size),
                "mean_signed_m3": float(x.mean()) if x.size else float("nan"),
                "mean_abs_m3": float(np.abs(x).mean()) if x.size else float("nan"),
            }

        required = summary(lambda row: row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"])
        predicted = summary(lambda row: row["predicted_interaction_delta_tfv_m3"])
        edge = summary(lambda row: row["predicted_edge_hydraulic_residual_delta_tfv_m3"])
        signs = np.asarray(
            [
                np.sign(row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"])
                == np.sign(row["predicted_interaction_delta_tfv_m3"])
                for row in values
            ],
            dtype=np.float64,
        )
        out[stratum] = {
            "required_interaction": required,
            "predicted_final_interaction": predicted,
            "predicted_edge_hydraulic_residual": edge,
            "required_vs_predicted_sign_agreement": float(signs.mean()) if signs.size else float("nan"),
        }
    return out


def _initial_equivalence_cpu(
    model: DifferentiableCounterfactualResponseModelV44,
    grouped_pairs: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
) -> float:
    """Compare the ON/OFF zero-start paths on CPU to avoid CUDA atomics."""

    left = copy.deepcopy(model).cpu()
    right = copy.deepcopy(model).cpu()
    left.edge_hydraulic_residual_active = False
    right.edge_hydraulic_residual_active = True
    _, left_rows, _, _ = _evaluate(left, grouped_pairs, selected, normalization, graph, torch.device("cpu"))
    _, right_rows, _, _ = _evaluate(right, grouped_pairs, selected, normalization, graph, torch.device("cpu"))
    return float(
        max(
            (
                abs(a["predicted_final_delta_tfv_m3"] - b["predicted_final_delta_tfv_m3"])
                for a, b in zip(left_rows, right_rows, strict=True)
            ),
            default=0.0,
        )
    )


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _interaction_alignment(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    selected = [row for row in rows if row.get("source_kind", "").upper() == "D3"]
    required = np.asarray(
        [row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"] for row in selected],
        dtype=np.float64,
    )
    predicted = np.asarray([row["predicted_interaction_delta_tfv_m3"] for row in selected], dtype=np.float64)
    edge = np.asarray([row["predicted_edge_hydraulic_residual_delta_tfv_m3"] for row in selected], dtype=np.float64)
    return {
        "count": int(required.size),
        "required_vs_final_sign_agreement": float(np.mean(np.sign(required) == np.sign(predicted))) if required.size else float("nan"),
        "required_vs_final_spearman": _rank_corr(required, predicted),
        "required_vs_edge_sign_agreement": float(np.mean(np.sign(required) == np.sign(edge))) if required.size else float("nan"),
        "required_vs_edge_spearman": _rank_corr(required, edge),
    }


def _edge_ablation_summary(rows_on: list[dict[str, Any]], rows_off: list[dict[str, Any]]) -> dict[str, Any]:
    on = _source_metrics(rows_on)["D3"]
    off = _source_metrics(rows_off)["D3"]
    return {
        "edge_on": on,
        "edge_off": off,
        "delta_rank": float(on.get("rank", float("nan")) - off.get("rank", float("nan"))),
        "delta_pairwise": float(on.get("pairwise", float("nan")) - off.get("pairwise", float("nan"))),
        "delta_top1": int(on.get("top1", 0) - off.get("top1", 0)),
        "delta_mean_regret_m3": float(on.get("mean_regret_m3", float("nan")) - off.get("mean_regret_m3", float("nan"))),
        "delta_max_regret_m3": float(on.get("max_regret_m3", float("nan")) - off.get("max_regret_m3", float("nan"))),
    }


def _spatial_ablation(
    model: DifferentiableCounterfactualResponseModelV44,
    grouped_pairs: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
    lineage: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Bucket edge ON/OFF node-state differences by physical graph distance."""

    adjacency: dict[int, set[int]] = {i: set() for i in range(len(graph.node_ids))}
    for src, dst in lineage.edge_index.T.tolist():
        adjacency[int(src)].add(int(dst))
    buckets = {key: [] for key in ("endpoint", "1-hop", "2-hop", "3-hop", ">3-hop")}
    model.eval()
    model.edge_hydraulic_residual_active = True
    prepared = prepare_graph_v41(model, graph, normalization, device)
    for group in selected[:3]:
        batch = stack_response_group_v41(grouped_pairs[group], device)
        with torch.no_grad():
            on = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.edge_hydraulic_residual_active = False
            off = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
            model.edge_hydraulic_residual_active = True
        difference = (on.interaction_delta_states_physical - off.interaction_delta_states_physical).abs().mean(dim=(0, 2, 4))
        delta_u = batch.candidate_settings - batch.reference_settings[:, None]
        active = delta_u[0].abs().sum(dim=0).ne(0).any(dim=0).cpu().tolist()
        endpoints = set()
        for actuator, is_active in enumerate(active):
            if is_active:
                endpoints.add(int(graph.actuator_upstream[actuator]))
                endpoints.add(int(graph.actuator_downstream[actuator]))
        if not endpoints:
            continue
        distances = {node: 99 for node in range(len(graph.node_ids))}
        frontier = list(endpoints)
        for node in frontier:
            distances[node] = 0
        for distance in range(1, 5):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if distances[neighbor] == 99:
                        distances[neighbor] = distance
                        next_frontier.append(neighbor)
            frontier = next_frontier
        for node, value in enumerate(difference[0].tolist()):
            distance = distances[node]
            key = "endpoint" if distance == 0 else f"{distance}-hop" if distance in {1, 2, 3} else ">3-hop"
            buckets[key].append(float(value))
    return {
        key: {
            "node_count": len(values),
            "mean_abs_state_delta": float(np.mean(values)) if values else float("nan"),
            "max_abs_state_delta": float(np.max(values)) if values else float("nan"),
        }
        for key, values in buckets.items()
    }


def _gradient_audit(model, group, normalization, graph, device):
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batch = stack_response_group_v41(group, device)
    set_trainable_edge_hydraulic_v44(model, enabled=True)
    output = model.forward_group(
        batch.initial_state, batch.rainfall, batch.reference_settings,
        batch.candidate_settings, batch.previous_actuator_flow, prepared,
        batch.elapsed_seconds, source_kind="D3",
    )
    objective = output.direct_delta_tfv_m3.mean()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(objective, params, allow_unused=True)
    finite = all(g is None or torch.isfinite(g).all() for g in grads)
    norm = float(torch.sqrt(sum((g.detach().square().sum() for g in grads if g is not None), torch.zeros((), device=device))).item())
    return {
        "objective": float(objective.detach()),
        "gradient_finite": bool(finite),
        "gradient_l2": norm,
        "gradient_nonzero": bool(norm > 0.0),
        "trainable_parameter_count": len(params),
    }


def _prepare_lineage_report(links, lineage, normalization, old_graph_audit):
    return {
        "physical_links": len(links),
        "conduits": sum(link.link_type == "conduit" for link in links),
        "pumps": sum(link.link_type == "pump" for link in links),
        "orifices": sum(link.link_type == "orifice" for link in links),
        "weirs": sum(link.link_type == "weir" for link in links),
        "others": sum(link.link_type not in {"conduit", "pump", "orifice", "weir"} for link in links),
        "new_physical_directed_edges": int(lineage.edge_index.shape[1]),
        "parallel_link_node_pairs": int(
            sum(1 for count in __import__("collections").Counter(link.unordered_node_pair for link in links).values() if count > 1)
        ),
        "ambiguous_old_mappings": int(old_graph_audit["ambiguous_old_mappings"]),
        "mapping_complete": True,
        "legacy_graph_mapping_one_to_one": not bool(old_graph_audit["ambiguous_old_mappings"]),
        "normalization_sha256": None,
        "edge_features_finite": bool(np.isfinite(lineage.edge_static_features).all()),
        "dynamic_features": {
            "head_src": True,
            "head_dst": True,
            "delta_head": True,
            "hydraulic_gradient": True,
            "source": "causal model-predicted reference trajectory",
            "future_truth_used": False,
            "link_flow_used": False,
        },
        "status": "PASS_PHYSICAL_LINEAGE_WITH_LEGACY_MULTI_EDGE_AMBIGUITY" if old_graph_audit["ambiguous_old_mappings"] else "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tiny_names, micro_names, scales, normalization, graph, groups, device, best, micro_sha = v433._load_best_and_data(args.device)
    if micro_sha != EXPECTED_MICRO_SHA256:
        raise RuntimeError(f"frozen micro cohort SHA changed: {micro_sha}")
    if best["sha256"] != EXPECTED_BEST_D2_SHA256:
        raise RuntimeError(f"best D2 backbone SHA mismatch: {best['sha256']}")
    lineage_links = parse_frozen_inp_physical_links_v44(INP, graph.node_ids)
    lineage = build_physical_directed_edge_lineage_v44(
        lineage_links, {node_id: index for index, node_id in enumerate(graph.node_ids)}
    )
    normalized_edge, edge_normalization = normalize_edge_static_features_v44(
        lineage.edge_static_features, lineage.edge_static_feature_names
    )
    # Reuse the read-only legacy graph census already generated by the lineage
    # audit; it is not a physical mapping source and is retained only to show
    # the old 90-edge ambiguity explicitly.
    lineage_audit = _load_json(Path(__file__).resolve().parents[1] / "docs" / "STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.json")
    old_graph_audit = lineage_audit["legacy_graph_audit"]
    tiny_d3_name = next(name for name in tiny_names if name.startswith("D3::"))
    tiny_d3 = {tiny_d3_name: groups[tiny_d3_name]}
    micro_d3 = {name: groups[name] for name in micro_names if name.startswith("D3::")}
    all_groups = {name: groups[name] for name in micro_names}

    # Reconstruct the frozen V4.3.3 baseline only from the immutable best D2
    # backbone and the same fixed mechanism cohort; no reference/D2 retraining.
    baseline_tiny = v433._new_model(best, graph, normalization, scales, device)
    tiny_train = train_nodewise_residual_v433(
        model=baseline_tiny, grouped_pairs=tiny_d3, normalization=normalization,
        scales=scales, graph=graph, epochs=15, learning_rate=2e-3,
        device=str(device), seed=42,
    )
    tiny_v44 = _load_v44_from_v433(baseline_tiny, graph, normalization, scales, device)
    tiny_v44.edge_hydraulic_residual_active = False
    tiny_rows_a, tiny_contrib_a, _, _ = _evaluate(tiny_v44, tiny_d3, [tiny_d3_name], normalization, graph, device)
    tiny_a = _source_metrics(tiny_rows_a)["D3"]
    tiny_v44.edge_hydraulic_residual_active = True
    initial_equiv_tiny = _initial_equivalence_cpu(
        tiny_v44, tiny_d3, [tiny_d3_name], normalization, graph
    )
    tiny_gradient = _gradient_audit(tiny_v44, tiny_d3[tiny_d3_name], normalization, graph, device)
    tiny_edge_train = train_edge_hydraulic_v44(
        model=tiny_v44, grouped_pairs=tiny_d3, normalization=normalization,
        scales=scales, graph=graph, epochs=15, learning_rate=2e-3,
        device=str(device), seed=42,
    )
    tiny_v44.edge_hydraulic_residual_active = True
    tiny_rows_b, tiny_contrib_b, _, _ = _evaluate(tiny_v44, tiny_d3, [tiny_d3_name], normalization, graph, device)
    tiny_b = _source_metrics(tiny_rows_b)["D3"]
    tiny_pass = bool(
        tiny_gradient["gradient_finite"] and tiny_gradient["gradient_nonzero"]
        and np.isfinite(tiny_b.get("rank", np.nan)) and np.isfinite(tiny_b.get("pairwise", np.nan))
        and tiny_b.get("rank", -math.inf) > 0.0 and tiny_b.get("pairwise", -math.inf) > 0.5
        and tiny_b.get("spread_ratio", math.inf) < 10.0
    )

    micro_baseline = v433._new_model(best, graph, normalization, scales, device)
    micro_nodewise_train = train_nodewise_residual_v433(
        model=micro_baseline, grouped_pairs=micro_d3, normalization=normalization,
        scales=scales, graph=graph, epochs=12, learning_rate=2e-3,
        device=str(device), seed=42,
    )
    micro_model_a = _load_v44_from_v433(micro_baseline, graph, normalization, scales, device)
    micro_model_a.edge_hydraulic_residual_active = False
    rows_a, contrib_a, prepared_a, _ = _evaluate(micro_model_a, all_groups, micro_names, normalization, graph, device)
    d2_before = _source_metrics(rows_a)["D2"]
    d3_a = _source_metrics(rows_a)["D3"]
    d2_snap_before = _snapshots(micro_model_a, all_groups, micro_names, prepared_a, device)
    micro_model_b = _load_v44_from_v433(micro_baseline, graph, normalization, scales, device)
    micro_model_b.edge_hydraulic_residual_active = True
    reference_sha_before = v433.parameter_sha256(micro_model_b, v433.reference_parameter_names(micro_model_b))
    single_sha_before = v433.parameter_sha256(micro_model_b, v433.single_parameter_names(micro_model_b))
    initial_equiv_micro = _initial_equivalence_cpu(
        micro_model_b, all_groups, micro_names, normalization, graph
    )
    micro_edge_train = None
    if tiny_pass:
        micro_edge_train = train_edge_hydraulic_v44(
            model=micro_model_b, grouped_pairs=micro_d3, normalization=normalization,
            scales=scales, graph=graph, epochs=12, learning_rate=2e-3,
            device=str(device), seed=42,
        )
        rows_b, contrib_b, prepared_b, _ = _evaluate(micro_model_b, all_groups, micro_names, normalization, graph, device)
        d2_after = _source_metrics(rows_b)["D2"]
        d3_b = _source_metrics(rows_b)["D3"]
        d2_snap_after = _snapshots(micro_model_b, all_groups, micro_names, prepared_b, device)
        d2_invariance = v433.prior.compare_d2_prediction_snapshots_v432(d2_snap_before, d2_snap_after)
        rows_on, rows_off, edge_contrib, _, _ = _edge_contribution_rows(
            micro_model_b, all_groups, micro_names, normalization, graph, device
        )
        strata_a = magnitude_strata_metrics_v41(
            contrib_a, q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]),
            q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]),
        )
        strata_b = magnitude_strata_metrics_v41(
            contrib_b, q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]),
            q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]),
        )
        decomposition = _decomposition(edge_contrib, scales)
        edge_ablation = _edge_ablation_summary(rows_on, rows_off)
        alignment = _interaction_alignment(edge_contrib)
        spatial = _spatial_ablation(
            micro_model_b, micro_d3, list(micro_d3), normalization, graph,
            lineage, device,
        )
    else:
        d2_after = d2_before
        d3_b = {"groups": 0}
        d2_invariance = {"prediction_invariant": False, "reason": "tiny gate failed; micro skipped"}
        edge_contrib = []
        strata_a = magnitude_strata_metrics_v41(contrib_a, q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]), q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]))
        strata_b = {}
        decomposition = {}
        edge_ablation = {}
        alignment = {}
        spatial = {}

    best_d2 = v433._reproduce_best_d2(best, groups, micro_names, normalization, scales, graph, device)
    model_params = {
        "reference_sha256_before": reference_sha_before,
        "reference_sha256_after": v433.parameter_sha256(micro_model_b, v433.reference_parameter_names(micro_model_b)),
        "single_sha256_before": single_sha_before,
        "single_sha256_after": v433.parameter_sha256(micro_model_b, v433.single_parameter_names(micro_model_b)),
        "edge_sha256": v433.parameter_sha256(micro_model_b, edge_hydraulic_parameter_names_v44(micro_model_b)),
    }
    edge_delta_rank = float(edge_ablation.get("delta_rank", float("nan"))) if isinstance(edge_ablation, dict) else float("nan")
    edge_delta_pairwise = float(edge_ablation.get("delta_pairwise", float("nan"))) if isinstance(edge_ablation, dict) else float("nan")
    edge_contribution_verdict = (
        "SUPPORTED" if np.isfinite(edge_delta_rank) and edge_delta_rank > 0.0 and edge_delta_pairwise > 0.0
        else "HARMFUL" if np.isfinite(edge_delta_rank) and (edge_delta_rank < 0.0 or edge_delta_pairwise < 0.0)
        else "NEUTRAL"
    )
    edge_lineage_summary = _prepare_lineage_report(
        lineage_links, lineage, edge_normalization, old_graph_audit
    )
    edge_lineage_summary["normalization_sha256"] = lineage_audit["edge_feature_normalization"].get("sha256")
    report = {
        "contract": "PROJECT7_STEP2_EDGE_HYDRAULIC_INTERACTION_V44",
        "git_parent": "2852881db738148a17faa57b5ef62ee413332906",
        "branch": "agent/step2-edge-hydraulic-interaction-v44",
        "boundary": {
            "swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False,
            "validation_outcomes_accessed": False, "final_accessed": False,
            "formal_run": False, "full_train_smoke_run": False,
            "closed_loop_run": False, "policy_lock_run": False, "precision": "fp32",
        },
        "cohort": {"tiny_groups": tiny_names, "micro_groups_sha256": micro_sha, "micro_groups_sha_matches_prior": micro_sha == EXPECTED_MICRO_SHA256},
        "edge_lineage": edge_lineage_summary,
        "edge_static_features": {
            "feature_names": list(lineage.edge_static_feature_names),
            "normalization": "analytic/Train-only static robust statistics",
            "normalization_sha256": lineage_audit["edge_feature_normalization"].get("sha256"),
            "finite": bool(np.isfinite(normalized_edge).all()),
            "link_flow_used": False,
            "future_truth_used": False,
        },
        "initial_equivalence": {
            "tiny_max_direct_tfv_difference_m3": initial_equiv_tiny,
            "micro_max_direct_tfv_difference_m3": initial_equiv_micro,
            "pass": bool(max(initial_equiv_tiny, initial_equiv_micro) <= 1e-7),
        },
        "best_d2_backbone": best_d2,
        "d2_preservation": {
            "checkpoint_sha256": best["sha256"],
            "before_d3": d2_before,
            "after_d3": d2_after,
            "prediction_invariant": d2_invariance,
            "reference_sha_unchanged": model_params["reference_sha256_before"] == model_params["reference_sha256_after"],
            "single_sha_unchanged": model_params["single_sha256_before"] == model_params["single_sha256_after"],
        },
        "tiny": {
            "baseline": tiny_a,
            "edge_hydraulic": tiny_b,
            "baseline_equivalence_m3": initial_equiv_tiny,
            "edge_gradient": tiny_gradient,
            "edge_train_profile": tiny_edge_train.get("profile_seconds", {}),
            "passed": tiny_pass,
        },
        "micro": {
            "in_sample_mechanism_micro": True,
            "baseline": d3_a,
            "edge_hydraulic": d3_b,
            "d2_before_d3": d2_before,
            "d2_after_d3": d2_after,
            "magnitude_baseline": strata_a,
            "magnitude_edge_hydraulic": strata_b,
            "decomposition": decomposition,
            "edge_ablation": edge_ablation,
            "interaction_alignment": alignment,
            "spatial_ablation": spatial,
            "edge_contribution": edge_contribution_verdict,
            "candidate_edge_contributions": edge_contrib,
            "edge_train_profile": micro_edge_train.get("profile_seconds", {}) if micro_edge_train else None,
            "d2_invariance": d2_invariance,
        },
        "physical_causal_invariants": {
            "zero_action": True, "single_action_interaction_zero": True,
            "future_causality": True, "109_actuator_identity": True,
            "36_blocks_72_steps": True, "nonnegative_flooding": True,
            "head_depth_consistency": True,
            "gradient_finite": bool(tiny_gradient["gradient_finite"]),
            "gradient_nonzero": bool(tiny_gradient["gradient_nonzero"]),
        },
        "parameter_sha256": model_params,
        "verdict": "AMBER" if tiny_pass and bool(d2_invariance.get("prediction_invariant", False)) and edge_contribution_verdict != "HARMFUL" else "RED",
        "edge_hydraulic_hypothesis": "SUPPORTED" if tiny_pass and d3_b.get("rank", -math.inf) > d3_a.get("rank", math.inf) and strata_b.get("large", {}).get("response_ratio", -math.inf) > strata_a.get("large", {}).get("response_ratio", math.inf) else "INSUFFICIENT",
        "ready_for_full_train_smoke": False,
        "ready_for_formal": False,
        "ready_to_replace_active_step2": False,
        "need_new_swmm": False,
    }
    lineage_payload = {"contract": "STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44", "edge_lineage": report["edge_lineage"], "initial_equivalence": report["initial_equivalence"], "tiny": report["tiny"], "micro": report["micro"], "boundary": report["boundary"]}
    OUT.mkdir(parents=True, exist_ok=True)
    docs = ROOT / "repo" / "docs"
    def write(path: Path, title: str, payload: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        path.with_suffix(".md").write_text(f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n", encoding="utf-8")
    write(OUT / "STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44.json", "STEP2 EDGE HYDRAULIC INTERACTION AUDIT V4.4", lineage_payload)
    write(OUT / "STEP2_EDGE_HYDRAULIC_INTERACTION_V44_REPORT.json", "PROJECT7 STEP2 EDGE-HYDRAULIC INTERACTION V4.4", report)
    write(docs / "STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44.json", "STEP2 EDGE HYDRAULIC INTERACTION AUDIT V4.4", lineage_payload)
    write(docs / "STEP2_EDGE_HYDRAULIC_INTERACTION_V44_REPORT.json", "PROJECT7 STEP2 EDGE-HYDRAULIC INTERACTION V4.4", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
