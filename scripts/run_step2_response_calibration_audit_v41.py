"""Run the frozen Train-only Step2 V4.1 response-calibration audit.

This script reads existing Project7 V4 checkpoints and development/train cache arrays.
It never launches SWMM, trains a model, reads Validation outcomes, or reads Final.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_control_response_v4 import DifferentiableCounterfactualResponseModelV4
from rtc.step2_response_calibration_audit_v41 import (
    _parameter_group_entries,
    cumulative_trapezoid_pair_delta_tfv,
    current_v4_loss_components,
    gradient_cosine,
    head_depth_consistency,
    magnitude_statistics,
    parameter_group_parameter_counts,
    reference_forward_accounting,
)
from rtc.step2_train_response_v4 import (
    _norm,
    _stack_pairs,
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_training_cache import load_step2_training_cache

CHANNELS = ("depth", "head", "flooding_rate", "volume", "inflow", "outflow")
ACTUATOR_TYPES = ("pump", "orifice", "weir", "outlet")
CURRENT_LOSS_WEIGHTS = {
    "absolute_state": 0.5,
    "absolute_flow": 0.5,
    "delta_state": 1.0,
    "delta_flow": 1.0,
    "delta_tfv_rate_rectangle": 2.0,
    "ranking_sign": 1.0,
    "physical_nonnegative_penalty": 0.1,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats_row(base: dict[str, object], values: np.ndarray) -> dict[str, object]:
    return {**base, **magnitude_statistics(values, zero_atol=1e-12)}


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def actuator_types(graph) -> np.ndarray:
    kinds = np.argmax(np.asarray(graph.actuator_physics)[:, :4], axis=1)
    return np.asarray([ACTUATOR_TYPES[int(index)] for index in kinds])


def read_group_column(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def model_for(graph, norm, checkpoint: Path, device: torch.device):
    model = DifferentiableCounterfactualResponseModelV4(
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=160,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        state_effect_scale=torch.as_tensor(
            norm.state_delta_scale / np.maximum(norm.state_std, 1e-6)
        ),
    ).to(device).float()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def stack_source(grouped_pairs, source: str) -> dict[str, Any]:
    records = [
        (group, pair)
        for group in sorted(grouped_pairs)
        if group.startswith(source + "::")
        for pair in grouped_pairs[group]
    ]
    if not records:
        raise ValueError(f"fixed audit cohort has no {source} candidates")
    groups = np.asarray([group for group, _pair in records])
    pairs = [pair for _group, pair in records]

    def delta(key: str) -> np.ndarray:
        return np.stack(
            [pair.candidate[key] - pair.reference[key] for pair in pairs], axis=0
        ).astype(np.float32, copy=False)

    reference_state = np.stack(
        [pair.reference["target_states_physical"] for pair in pairs], axis=0
    )
    candidate_state = np.stack(
        [pair.candidate["target_states_physical"] for pair in pairs], axis=0
    )
    reference_initial = np.stack(
        [pair.reference["initial_state_physical"] for pair in pairs], axis=0
    )
    candidate_initial = np.stack(
        [pair.candidate["initial_state_physical"] for pair in pairs], axis=0
    )
    elapsed = np.stack([pair.reference["elapsed_seconds"] for pair in pairs], axis=0)
    dt = np.diff(elapsed.astype(np.float64), axis=1)
    delta_rate = np.maximum(candidate_state[..., 2], 0.0) - np.maximum(
        reference_state[..., 2], 0.0
    )
    rectangle = np.cumsum((delta_rate * dt[:, :, None]).sum(axis=2), axis=1)
    trapezoid = cumulative_trapezoid_pair_delta_tfv(
        candidate_initial[..., 2],
        reference_initial[..., 2],
        candidate_state[..., 2],
        reference_state[..., 2],
        elapsed,
    )
    exact = np.asarray(
        [
            float(pair.candidate["exact_node_flood_volume_m3"].sum())
            - float(pair.reference["exact_node_flood_volume_m3"].sum())
            for pair in pairs
        ],
        dtype=np.float64,
    )
    settings = delta("settings")
    active_masks = np.any(np.abs(settings) > 1e-7, axis=1)
    return {
        "records": records,
        "pairs": pairs,
        "groups": groups,
        "delta_state": candidate_state - reference_state,
        "delta_flow": delta("target_actuator_flows_physical"),
        "delta_setting": settings,
        "candidate_state": candidate_state,
        "reference_state": reference_state,
        "elapsed": elapsed,
        "rectangle_tfv": rectangle,
        "trapezoid_tfv": trapezoid,
        "authoritative_tfv": exact,
        "active_masks": active_masks,
    }


def target_scale_tables(source_data: dict[str, dict[str, Any]], graph, norm):
    source_channel: list[dict[str, object]] = []
    source_horizon: list[dict[str, object]] = []
    source_node: list[dict[str, object]] = []
    source_actuator: list[dict[str, object]] = []
    group_channel: list[dict[str, object]] = []
    tfv_alignment: list[dict[str, object]] = []
    d2_actuator_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    kinds = actuator_types(graph)

    for source, data in source_data.items():
        state = data["delta_state"]
        flow = data["delta_flow"]
        setting = data["delta_setting"]
        rectangle = data["rectangle_tfv"]
        trapezoid = data["trapezoid_tfv"]
        exact = data["authoritative_tfv"]
        for channel, name in enumerate(CHANNELS):
            source_channel.append(
                stats_row(
                    {
                        "source_kind": source,
                        "target": f"delta_{name}",
                        "channel": channel,
                        "current_v4_absolute_std": float(norm.state_std[channel]),
                        "train_transition_delta_scale": float(norm.state_delta_scale[channel]),
                    },
                    state[..., channel],
                )
            )
        source_channel.append(
            stats_row(
                {
                    "source_kind": source,
                    "target": "delta_actuator_flow",
                    "channel": "all",
                    "current_v4_absolute_std": float(np.asarray(norm.flow_std).reshape(-1)[0]),
                    "train_transition_delta_scale": float(np.median(norm.flow_delta_scale)),
                },
                flow,
            )
        )
        for target, values in (
            ("delta_tfv_authoritative", exact),
            ("delta_tfv_rate_rectangle_current_v4", rectangle[:, -1]),
            ("delta_tfv_rate_trapezoid", trapezoid[:, -1]),
        ):
            source_channel.append(
                stats_row(
                    {
                        "source_kind": source,
                        "target": target,
                        "channel": "network",
                        "current_v4_absolute_std": "per_sample_abs_true_floor_1m3",
                        "train_transition_delta_scale": "not_defined",
                    },
                    values,
                )
            )

        for horizon in range(state.shape[1]):
            base = {
                "source_kind": source,
                "horizon_step": horizon + 1,
                "horizon_minutes": 5 * (horizon + 1),
            }
            for channel, name in enumerate(CHANNELS):
                source_horizon.append(
                    stats_row({**base, "target": f"delta_{name}", "channel": channel}, state[:, horizon, :, channel])
                )
            source_horizon.append(
                stats_row({**base, "target": "delta_actuator_flow", "channel": "all"}, flow[:, horizon])
            )
            source_horizon.append(
                stats_row({**base, "target": "delta_tfv_rate_rectangle_current_v4", "channel": "network"}, rectangle[:, horizon])
            )
            source_horizon.append(
                stats_row({**base, "target": "delta_tfv_rate_trapezoid", "channel": "network"}, trapezoid[:, horizon])
            )

        for node, node_id in enumerate(graph.node_ids):
            for channel, name in enumerate(CHANNELS):
                source_node.append(
                    stats_row(
                        {
                            "source_kind": source,
                            "node_id": node_id,
                            "target": f"delta_{name}",
                            "channel": channel,
                        },
                        state[:, :, node, channel],
                    )
                )
        for actuator, actuator_id in enumerate(graph.actuator_ids):
            common = {
                "source_kind": source,
                "actuator_id": actuator_id,
                "actuator_type": kinds[actuator],
            }
            source_actuator.append(
                stats_row({**common, "target": "delta_actuator_flow"}, flow[:, :, actuator])
            )
            source_actuator.append(
                stats_row({**common, "target": "delta_setting"}, setting[:, :, actuator])
            )

        for group in sorted(set(data["groups"].tolist())):
            mask = data["groups"] == group
            active_union = np.any(data["active_masks"][mask], axis=0)
            active_ids = [graph.actuator_ids[i] for i in np.flatnonzero(active_union)]
            common = {
                "group": group,
                "source_kind": source,
                "candidate_count": int(np.sum(mask)),
                "active_actuator_count": len(active_ids),
                "active_actuator_ids": "|".join(active_ids),
            }
            for channel, name in enumerate(CHANNELS):
                group_channel.append(
                    stats_row({**common, "target": f"delta_{name}", "channel": channel}, state[mask, ..., channel])
                )
            group_channel.append(
                stats_row({**common, "target": "delta_actuator_flow", "channel": "all"}, flow[mask])
            )
            group_channel.append(
                stats_row({**common, "target": "delta_tfv_authoritative", "channel": "network"}, exact[mask])
            )
            exact_group = exact[mask]
            rectangle_group = rectangle[mask, -1]
            trapezoid_group = trapezoid[mask, -1]
            tfv_alignment.append(
                {
                    **common,
                    "authoritative_spread_m3": float(np.ptp(exact_group)),
                    "rectangle_spread_m3": float(np.ptp(rectangle_group)),
                    "trapezoid_spread_m3": float(np.ptp(trapezoid_group)),
                    "rectangle_to_authoritative_spread_ratio": float(np.ptp(rectangle_group) / max(np.ptp(exact_group), 1e-12)),
                    "trapezoid_to_authoritative_spread_ratio": float(np.ptp(trapezoid_group) / max(np.ptp(exact_group), 1e-12)),
                    "rectangle_authoritative_spearman": spearman(rectangle_group, exact_group),
                    "trapezoid_authoritative_spearman": spearman(trapezoid_group, exact_group),
                    "rectangle_authoritative_sign_accuracy": float(np.mean(np.sign(rectangle_group) == np.sign(exact_group))),
                    "trapezoid_authoritative_sign_accuracy": float(np.mean(np.sign(trapezoid_group) == np.sign(exact_group))),
                    "rectangle_top1": bool(np.argmin(rectangle_group) == np.argmin(exact_group)),
                    "trapezoid_top1": bool(np.argmin(trapezoid_group) == np.argmin(exact_group)),
                }
            )

    d2 = source_data["D2"]
    for actuator, actuator_id in enumerate(graph.actuator_ids):
        candidate_mask = d2["active_masks"][:, actuator]
        if not np.any(candidate_mask):
            continue
        active_counts = d2["active_masks"][candidate_mask].sum(axis=1)
        if not np.all(active_counts == 1):
            raise ValueError("D2 active-actuator attribution found a multi-actuator candidate")
        exact = d2["authoritative_tfv"][candidate_mask]
        rectangle = d2["rectangle_tfv"][candidate_mask, -1]
        d2_actuator_rows.append(
            {
                "actuator_id": actuator_id,
                "actuator_type": kinds[actuator],
                "candidate_count": int(np.sum(candidate_mask)),
                "authoritative_delta_tfv_rms_m3": magnitude_statistics(exact)["rms"],
                "authoritative_delta_tfv_median_abs_m3": magnitude_statistics(exact)["median_abs"],
                "authoritative_delta_tfv_max_abs_m3": magnitude_statistics(exact)["max_abs"],
                "current_rectangle_sign_accuracy": float(np.mean(np.sign(rectangle) == np.sign(exact))),
                "active_actuator_flow_delta_rms_m3s": magnitude_statistics(d2["delta_flow"][candidate_mask, :, actuator])["rms"],
            }
        )
    for kind in ACTUATOR_TYPES:
        rows = [row for row in d2_actuator_rows if row["actuator_type"] == kind]
        if not rows:
            continue
        indices = [graph.actuator_ids.index(str(row["actuator_id"])) for row in rows]
        candidate_mask = np.any(d2["active_masks"][:, indices], axis=1)
        exact = d2["authoritative_tfv"][candidate_mask]
        rectangle = d2["rectangle_tfv"][candidate_mask, -1]
        type_rows.append(
            {
                "actuator_type": kind,
                "actuator_identity_count": len(rows),
                "candidate_count": int(np.sum(candidate_mask)),
                **{f"authoritative_delta_tfv_{key}": value for key, value in magnitude_statistics(exact).items()},
                "current_rectangle_sign_accuracy": float(np.mean(np.sign(rectangle) == np.sign(exact))),
            }
        )
    return {
        "source_channel": source_channel,
        "source_horizon": source_horizon,
        "source_node": source_node,
        "source_actuator": source_actuator,
        "group_channel": group_channel,
        "tfv_alignment": tfv_alignment,
        "d2_active_actuator": d2_actuator_rows,
        "d2_actuator_type": type_rows,
    }


def flooding_semantics_audit(source_data: dict[str, dict[str, Any]], graph) -> dict[str, object]:
    names = list(graph.static_node_feature_names)
    required = ("invert_elevation_m", "max_depth_m", "surcharge_depth_m", "ponded_area_m2")
    present = {name: name in names for name in required}
    rows = []
    for source, data in source_data.items():
        states = data["candidate_state"]
        depth = states[..., 0]
        flood = states[..., 2] > 1e-9
        max_depth = graph.static_node_features[:, names.index("max_depth_m")]
        surcharge = graph.static_node_features[:, names.index("surcharge_depth_m")]
        hypotheses = {
            "depth_gt_max_depth": depth > (max_depth.reshape(1, 1, -1) + 1e-6),
            "depth_gt_max_plus_surcharge": depth > ((max_depth + surcharge).reshape(1, 1, -1) + 1e-6),
        }
        for name, predicted in hypotheses.items():
            tp = int(np.sum(predicted & flood))
            fp = int(np.sum(predicted & ~flood))
            fn = int(np.sum(~predicted & flood))
            tn = int(np.sum(~predicted & ~flood))
            rows.append(
                {
                    "source_kind": source,
                    "hypothesis": name,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_negative": tn,
                    "agreement": float((tp + tn) / max(tp + fp + fn + tn, 1)),
                    "precision": float(tp / max(tp + fp, 1)),
                    "recall": float(tp / max(tp + fn, 1)),
                }
            )
    return {
        "static_features_present": present,
        "candidate_threshold_hypotheses": rows,
        "unambiguous_occurrence_gate_authorized": False,
        "reason": "The frozen graph exposes candidate depth thresholds, but neither tested threshold is a universal SWMM flooding-occurrence identity across node types and ponding/surcharge semantics. V4.1 may enforce non-negative physical flooding rate only; no guessed occurrence gate is authorized.",
    }


def gradient_group_vector(model, gradient_map: dict[str, torch.Tensor], group: str) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name, parameter, rows in _parameter_group_entries(model)[group]:
        gradient = gradient_map.get(name)
        if gradient is None:
            gradient = torch.zeros_like(parameter, device="cpu")
        if rows is not None:
            gradient = gradient[list(rows)]
        pieces.append(gradient.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def loss_gradient_audit(model, grouped_pairs, norm, graph, device: torch.device, batch_size: int = 4):
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    static = torch.as_tensor(_norm(graph.static_node_features, norm.static_mean, norm.static_std), device=device)
    physics = torch.as_tensor(_norm(graph.actuator_physics, norm.physics_mean, norm.physics_std), device=device)
    up = torch.as_tensor(graph.actuator_upstream, device=device)
    down = torch.as_tensor(graph.actuator_downstream, device=device)
    edges = torch.as_tensor(graph.edge_index, device=device)
    by_source: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    losses: dict[str, dict[str, float]] = {}

    for source in ("D2", "D3"):
        pairs = [
            pair
            for group in sorted(grouped_pairs)
            if group.startswith(source + "::")
            for pair in grouped_pairs[group]
        ]
        accum: dict[str, dict[str, torch.Tensor]] = {}
        value_accum: dict[str, float] = {}
        for start in range(0, len(pairs), batch_size):
            selected = pairs[start : start + batch_size]
            batch, reference_settings, candidate_settings = _stack_pairs(selected, device, norm)
            output = model(
                batch["initial_state"],
                batch["rainfall"],
                reference_settings,
                candidate_settings,
                batch["previous_actuator_flow"],
                static,
                physics,
                up,
                down,
                edges,
            )
            components, _diagnostic = current_v4_loss_components(output, batch, norm)
            names = list(components)
            weight = len(selected) / len(pairs)
            for index, component in enumerate(names):
                gradients = torch.autograd.grad(
                    components[component],
                    [parameter for _name, parameter in parameters],
                    retain_graph=index < len(names) - 1,
                    allow_unused=True,
                )
                target = accum.setdefault(component, {})
                for (parameter_name, parameter), gradient in zip(parameters, gradients):
                    value = torch.zeros_like(parameter, device="cpu") if gradient is None else gradient.detach().cpu()
                    target[parameter_name] = target.get(parameter_name, torch.zeros_like(value)) + value * weight
                value_accum[component] = value_accum.get(component, 0.0) + float(components[component].detach()) * weight
        weighted: dict[str, torch.Tensor] = {}
        for component, component_weight in CURRENT_LOSS_WEIGHTS.items():
            for parameter_name, gradient in accum[component].items():
                weighted[parameter_name] = weighted.get(parameter_name, torch.zeros_like(gradient)) + gradient * component_weight
        accum["weighted_total_current_v4"] = weighted
        value_accum["weighted_total_current_v4"] = sum(
            CURRENT_LOSS_WEIGHTS[name] * value_accum[name] for name in CURRENT_LOSS_WEIGHTS
        )
        by_source[source] = accum
        losses[source] = value_accum

    norm_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    cross_source_rows: list[dict[str, object]] = []
    groups = list(parameter_group_parameter_counts(model))
    for source, components in by_source.items():
        for group in groups:
            vectors = {name: gradient_group_vector(model, gradients, group) for name, gradients in components.items()}
            for component, vector in vectors.items():
                norm_rows.append(
                    {
                        "source_kind": source,
                        "loss_component": component,
                        "parameter_group": group,
                        "parameter_count": int(vector.size),
                        "gradient_norm": float(np.linalg.norm(vector)),
                        "gradient_finite_fraction": float(np.mean(np.isfinite(vector))) if vector.size else float("nan"),
                        "gradient_nonzero_fraction": float(np.mean(np.abs(vector) > 0.0)) if vector.size else float("nan"),
                        "loss_value": losses[source][component],
                    }
                )
            names = list(vectors)
            for i, left in enumerate(names):
                for right in names[i:]:
                    cosine_rows.append(
                        {
                            "source_kind": source,
                            "parameter_group": group,
                            "loss_left": left,
                            "loss_right": right,
                            "gradient_cosine": gradient_cosine(vectors[left], vectors[right]) if vectors[left].size else float("nan"),
                        }
                    )
    for component in by_source["D2"]:
        for group in groups:
            left = gradient_group_vector(model, by_source["D2"][component], group)
            right = gradient_group_vector(model, by_source["D3"][component], group)
            cross_source_rows.append(
                {
                    "loss_component": component,
                    "parameter_group": group,
                    "d2_d3_gradient_cosine": gradient_cosine(left, right) if left.size else float("nan"),
                    "d2_gradient_norm": float(np.linalg.norm(left)),
                    "d3_gradient_norm": float(np.linalg.norm(right)),
                    "d3_to_d2_norm_ratio": float(np.linalg.norm(right) / max(np.linalg.norm(left), 1e-30)),
                }
            )
    return norm_rows, cosine_rows, cross_source_rows, losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    repo = Path(args.repo).resolve()
    study = Path(args.study_root).resolve()
    out = Path(args.out_dir).resolve()
    v4 = study / "step2_control_response_v4"
    graph_path = study / "formal_assets" / "graph_schema.npz"
    cache_dir = study / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
    cache_manifest = cache_dir / "CACHE_MANIFEST.json"
    v3_model = study / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
    scale_json = study / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
    current_checkpoint = v4 / "03_12_group_micro" / "03_12_group_micro.pt"
    tiny_checkpoint = v4 / "02_tiny_overfit" / "02_tiny_overfit.pt"
    tiny_groups = read_group_column(v4 / "02_tiny_overfit" / "02_tiny_overfit.groups.csv")
    selected_groups = read_group_column(v4 / "03_12_group_micro" / "03_12_group_micro.groups.csv")
    if len(tiny_groups) != 2 or len(selected_groups) != 12:
        raise RuntimeError("V4.1 audit requires the frozen 2-group tiny and 12-group micro cohorts")
    if not set(tiny_groups).issubset(selected_groups):
        raise RuntimeError("frozen tiny groups are not contained in the frozen 12-group cohort")
    if sum(group.startswith("D2::") for group in selected_groups) != 6 or sum(
        group.startswith("D3::") for group in selected_groups
    ) != 6:
        raise RuntimeError("frozen micro cohort must contain six D2 and six D3 groups")

    raw_cache = load_step2_training_cache(cache_manifest)
    split_values = set()
    fold_values = set()
    for item in raw_cache["shards"]:
        split_values.update(str(value).lower() for value in item["arrays"]["scientific_split"].tolist())
        fold_values.update(str(value).lower() for value in item["arrays"]["development_fold"].tolist())
    if split_values != {"development"} or fold_values != {"train"}:
        raise RuntimeError(f"audit cache is not development/train only: {split_values=} {fold_values=}")

    graph = _load_graph(graph_path)
    norm = build_full_train_normalization_from_checkpoint(v3_model, scale_json)
    grouped_pairs = load_train_groups(cache_dir, norm, selected_groups)
    source_data = {source: stack_source(grouped_pairs, source) for source in ("D2", "D3")}
    data_loaded = time.perf_counter()
    tables = target_scale_tables(source_data, graph, norm)
    target_stats_done = time.perf_counter()

    invert_index = list(graph.static_node_feature_names).index("invert_elevation_m")
    invert = graph.static_node_features[:, invert_index]
    head_depth = {
        source: head_depth_consistency(data["candidate_state"], invert)
        for source, data in source_data.items()
    }
    flooding_semantics = flooding_semantics_audit(source_data, graph)

    requested_device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model_for(graph, norm, current_checkpoint, requested_device)
    norms, cosines, cross_source, loss_values = loss_gradient_audit(
        model, grouped_pairs, norm, graph, requested_device, batch_size=4
    )
    gradients_done = time.perf_counter()

    write_csv(out / "target_scale_by_source_channel.csv", tables["source_channel"])
    write_csv(out / "target_scale_by_source_horizon_channel.csv", tables["source_horizon"])
    write_csv(out / "target_scale_by_source_node_channel.csv", tables["source_node"])
    write_csv(out / "target_scale_by_source_actuator.csv", tables["source_actuator"])
    write_csv(out / "target_scale_by_group_channel.csv", tables["group_channel"])
    write_csv(out / "tfv_target_alignment_by_group.csv", tables["tfv_alignment"])
    write_csv(out / "d2_truth_by_active_actuator.csv", tables["d2_active_actuator"])
    write_csv(out / "d2_truth_by_actuator_type.csv", tables["d2_actuator_type"])
    write_csv(out / "loss_component_gradient_norms.csv", norms)
    write_csv(out / "loss_component_gradient_cosines.csv", cosines)
    write_csv(out / "d2_d3_gradient_cosines.csv", cross_source)

    scale_payload = json.loads(scale_json.read_text(encoding="utf-8"))
    details = scale_payload.get("details", {})
    source_summary = {
        row["source_kind"] + ":" + str(row["target"]): row
        for row in tables["source_channel"]
    }
    alignment_summary = {
        source: {
            "mean_rectangle_authoritative_spearman": float(
                np.nanmean([
                    row["rectangle_authoritative_spearman"]
                    for row in tables["tfv_alignment"]
                    if row["source_kind"] == source
                ])
            ),
            "mean_rectangle_to_authoritative_spread_ratio": float(
                np.mean([
                    row["rectangle_to_authoritative_spread_ratio"]
                    for row in tables["tfv_alignment"]
                    if row["source_kind"] == source
                ])
            ),
            "top1_fraction": float(
                np.mean([
                    row["rectangle_top1"]
                    for row in tables["tfv_alignment"]
                    if row["source_kind"] == source
                ])
            ),
        }
        for source in ("D2", "D3")
    }

    effect_groups = {"action_effect_encoder", "trajectory_effect_head", "flooding_head"}
    current_components = set(CURRENT_LOSS_WEIGHTS)
    dominant_rows = [
        row
        for row in norms
        if row["parameter_group"] in effect_groups and row["loss_component"] in current_components
    ]
    dominant = max(dominant_rows, key=lambda row: float(row["gradient_norm"]))
    zero_push_candidates = [
        row
        for row in cosines
        if row["parameter_group"] in effect_groups
        and row["loss_left"] in current_components
        and row["loss_right"] == "effect_energy_diagnostic"
        and np.isfinite(float(row["gradient_cosine"]))
    ]
    zero_push = max(zero_push_candidates, key=lambda row: float(row["gradient_cosine"]))
    reference_accounting = {
        source: {
            "groups": int(sum(group.startswith(source + "::") for group in selected_groups)),
            "candidates": len(source_data[source]["pairs"]),
            "per_group": [
                {"group": group, **reference_forward_accounting(candidate_count=len(grouped_pairs[group]))}
                for group in selected_groups
                if group.startswith(source + "::")
            ],
        }
        for source in ("D2", "D3")
    }

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report = {
        "contract": "STEP2_RESPONSE_CALIBRATION_AUDIT_V41",
        "status": "AUDIT_COMPLETE_NO_TRAINING",
        "boundary": {
            "scientific_split": sorted(split_values),
            "development_fold": sorted(fold_values),
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_step2_run": False,
            "closed_loop_run": False,
            "new_model_trained": False,
            "acceptance_thresholds_changed": False,
        },
        "inputs": {
            "git_head": git_head,
            "current_v4_checkpoint": str(current_checkpoint),
            "current_v4_checkpoint_sha256": sha256_file(current_checkpoint),
            "tiny_v4_checkpoint": str(tiny_checkpoint),
            "tiny_v4_checkpoint_sha256": sha256_file(tiny_checkpoint),
            "cache_manifest": str(cache_manifest),
            "cache_manifest_sha256": sha256_file(cache_manifest),
            "full_train_manifest_sha256": norm.source_manifest_sha256,
            "graph_schema_sha256": sha256_file(graph_path),
            "selected_groups": selected_groups,
            "tiny_groups": tiny_groups,
            "device": str(requested_device),
            "precision": "fp32",
        },
        "cohort": {
            "D2_groups": 6,
            "D2_candidates": len(source_data["D2"]["pairs"]),
            "D3_groups": 6,
            "D3_candidates": len(source_data["D3"]["pairs"]),
            "selection_reused_without_reselection": True,
        },
        "current_v4_loss_implementation": {
            "delta_state_normalization": "absolute_state_std",
            "delta_flow_normalization": "absolute_flow_std",
            "loaded_state_delta_scale_used_by_delta_loss": False,
            "loaded_flow_delta_scale_used_by_delta_loss": False,
            "delta_tfv_target": "future flooding-rate rectangle integration, not authoritative cumulative SWMM node flood volume",
            "delta_tfv_normalization": "per-candidate abs surrogate effect with 1 m3 floor",
            "ranking": "independent candidate-vs-reference sign classification, not within-group candidate ranking",
            "group_averaging": False,
            "candidate_averaging": True,
            "D2_D3_shared_effect_head": True,
            "direct_delta_tfv_head_exists": False,
            "reference_repetition": reference_accounting,
            "reference_loss_repeated_by_candidate_count": True,
            "parameter_group_counts": parameter_group_parameter_counts(model),
        },
        "normalization": {
            "absolute_state_std": np.asarray(norm.state_std).tolist(),
            "absolute_flow_std": np.asarray(norm.flow_std).tolist(),
            "transition_state_delta_scale": np.asarray(norm.state_delta_scale).tolist(),
            "transition_flow_delta_scale": np.asarray(norm.flow_delta_scale).tolist(),
            "transition_state_floor": details.get("state_floor"),
            "transition_flow_floor": details.get("flow_floor"),
            "source_target_summary": source_summary,
            "D2_D3_scale_imbalance": {
                target: float(source_summary[f"D3:{target}"]["rms"] / max(float(source_summary[f"D2:{target}"]["rms"]), 1e-30))
                for target in ("delta_actuator_flow", "delta_depth", "delta_head", "delta_flooding_rate", "delta_volume", "delta_inflow", "delta_outflow", "delta_tfv_authoritative")
            },
        },
        "tfv_target_alignment": alignment_summary,
        "head_depth_consistency": head_depth,
        "flooding_occurrence_semantics": flooding_semantics,
        "loss_values": loss_values,
        "gradient_audit": {
            "dominant_current_loss_on_effect_parameters": dominant,
            "strongest_near_zero_effect_alignment": zero_push,
            "norm_table": "loss_component_gradient_norms.csv",
            "cosine_table": "loss_component_gradient_cosines.csv",
            "D2_D3_cosine_table": "d2_d3_gradient_cosines.csv",
        },
        "root_cause": {
            "action_pathway_exists": True,
            "primary_calibration_failure": "TARGET_AND_LOSS_MISALIGNMENT",
            "findings": [
                "Counterfactual state/flow effects are normalized by absolute-state/flow standard deviations; the loaded Train-only delta scales do not normalize the effect losses.",
                "The TFV loss and sign objective use a rectangle-integrated flooding-rate surrogate rather than authoritative cumulative SWMM delta TFV.",
                "The ranking term is not group-aware and never compares candidate pairs within a checkpoint group.",
                "Reference encoding/loss is repeated for every pair, with two reference forward rows per candidate.",
                "D2 single-actuator and D3 multi-actuator responses share one effect head with no additive-single versus interaction-residual separation.",
                "There is no independent direct delta-TFV head, and flooding rate is emitted as a free signed state channel.",
            ],
        },
        "timing_seconds": {
            "data_load": data_loaded - started,
            "target_scale_tables": target_stats_done - data_loaded,
            "gradient_audit": gradients_done - target_stats_done,
            "total": gradients_done - started,
        },
        "outputs": {
            "target_scale_by_source_channel": "target_scale_by_source_channel.csv",
            "target_scale_by_source_horizon_channel": "target_scale_by_source_horizon_channel.csv",
            "target_scale_by_source_node_channel": "target_scale_by_source_node_channel.csv",
            "target_scale_by_source_actuator": "target_scale_by_source_actuator.csv",
            "target_scale_by_group_channel": "target_scale_by_group_channel.csv",
            "tfv_target_alignment_by_group": "tfv_target_alignment_by_group.csv",
            "d2_truth_by_active_actuator": "d2_truth_by_active_actuator.csv",
            "d2_truth_by_actuator_type": "d2_truth_by_actuator_type.csv",
        },
    }
    report_json = out / "STEP2_RESPONSE_CALIBRATION_AUDIT_V41.json"
    report_md = out / "STEP2_RESPONSE_CALIBRATION_AUDIT_V41.md"
    write_json(report_json, report)
    d2 = alignment_summary["D2"]
    d3 = alignment_summary["D3"]
    markdown = [
        "# Step2 response-calibration audit V4.1",
        "",
        "Status: **AUDIT COMPLETE — NO TRAINING**",
        "",
        "## Boundary",
        "",
        "- Existing development/train tiny and 12-group cohorts only.",
        "- SWMM launched: **NO**; D2/D3 regenerated: **NO**.",
        "- Validation outcomes: **NOT ACCESSED**; Final: **NOT ACCESSED**.",
        "- Formal Step2 / closed-loop: **NOT RUN**.",
        "",
        "## Root cause",
        "",
        "The V4 action pathway is nonzero, but the response objective is miscalibrated. Counterfactual state and flow losses still use absolute-state/flow standard deviations, despite loading Train-only delta scales. The TFV loss uses a future-rate rectangle surrogate rather than authoritative cumulative SWMM delta TFV, and its ranking term is only a candidate-vs-reference sign loss.",
        "",
        f"- D2 rate-surrogate versus authoritative mean group Spearman: `{d2['mean_rectangle_authoritative_spearman']:.4f}`; mean spread ratio: `{d2['mean_rectangle_to_authoritative_spread_ratio']:.4f}`.",
        f"- D3 rate-surrogate versus authoritative mean group Spearman: `{d3['mean_rectangle_authoritative_spearman']:.4f}`; mean spread ratio: `{d3['mean_rectangle_to_authoritative_spread_ratio']:.4f}`.",
        f"- Dominant current loss on effect parameters: `{dominant['source_kind']} / {dominant['loss_component']} / {dominant['parameter_group']}` with gradient norm `{float(dominant['gradient_norm']):.6g}`.",
        f"- Strongest gradient alignment that reduces effect energy: `{zero_push['source_kind']} / {zero_push['loss_left']} / {zero_push['parameter_group']}` with cosine `{float(zero_push['gradient_cosine']):.4f}`.",
        "- Direct delta-TFV head: **MISSING**.",
        "- Group-aware candidate-pair ranking: **MISSING**.",
        "- D2 single-effect versus D3 interaction separation: **MISSING**.",
        "- Reference deduplication: **MISSING**; D2 currently evaluates 48 reference rows for a 24-candidate group.",
        "",
        "## Physical semantics",
        "",
        f"- Head-depth consistency D2 max residual: `{head_depth['D2']['max_abs_residual_m']:.6g} m`; D3: `{head_depth['D3']['max_abs_residual_m']:.6g} m`.",
        "- The graph contains invert elevation, maximum depth, surcharge depth and ponded area, but the fixed cohort does not establish a universal flooding-occurrence threshold. No differentiable occurrence gate is authorized from this audit.",
        "- V4.1 should enforce non-negative reference/candidate flooding rates structurally while retaining signed counterfactual delta flooding.",
        "",
        "## Decision",
        "",
        "Proceed only to the bounded V4.1 architecture/loss change on development/train. Do not run 12-group micro until D2 tiny, D3 tiny and combined tiny pass in order. Need new SWMM: **NO**.",
    ]
    report_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": report["contract"],
                "status": report["status"],
                "report_json": str(report_json),
                "report_md": str(report_md),
                "root_cause": report["root_cause"],
                "timing_seconds": report["timing_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
