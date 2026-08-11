"""Run bounded Train-only stages for Step2 response calibration V4.1.

The script never launches SWMM and is deliberately isolated from the active/formal
Step2 trainer.  Each later stage requires a PASS gate written by the preceding stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41
from rtc.step2_train_response_v4 import (
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    evaluate_response_groups_v41,
    group_metrics_v41,
    load_or_derive_train_only_scales_v41,
    prepare_graph_v41,
    stack_response_group_v41,
    train_response_v41,
)
from rtc.step2_training_cache import load_step2_training_cache

STAGE_LAYOUT = {
    "tiny-d2": ("02_tiny_d2", "v41_tiny_d2.pt"),
    "tiny-d3": ("03_tiny_d3", "v41_tiny_d3.pt"),
    "tiny-combined": ("04_tiny_combined", "v41_tiny_combined.pt"),
    "micro": ("05_12_group_micro", "v41_12_group_micro.pt"),
}


def _read_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _model(
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    *,
    hidden_dim: int,
) -> DifferentiableCounterfactualResponseModelV41:
    return DifferentiableCounterfactualResponseModelV41(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=hidden_dim,
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
    )


def _load_parent(model: DifferentiableCounterfactualResponseModelV41, path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required preceding-stage checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC":
        raise RuntimeError(f"preceding checkpoint has wrong contract: {path}")
    incoming = payload["model_state_dict"]
    current = model.state_dict()
    incompatible = [
        name for name, value in incoming.items() if name in current and value.shape != current[name].shape
    ]
    if incompatible and not all(name.startswith("interaction_") for name in incompatible):
        raise RuntimeError(f"non-interaction parent parameter shape changed: {incompatible}")
    compatible = {
        name: value
        for name, value in incoming.items()
        if name in current and value.shape == current[name].shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected or any(not name.startswith("interaction_") for name in missing):
        raise RuntimeError(
            f"parent checkpoint mismatch outside resettable D3 interaction branch: "
            f"{missing=} {unexpected=}"
        )


def _require_gate(path: Path, source: str | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"required preceding-stage gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("mechanism_gates", {})
    selected = gates.get(source, {}) if source else gates
    if selected.get("verdict") != "PASS":
        raise RuntimeError(f"preceding stage did not PASS; fail closed at {path}")


def _gradient_diagnostic(
    model: DifferentiableCounterfactualResponseModelV41,
    pairs: list[Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch = stack_response_group_v41(pairs, device)
    candidate = batch.candidate_settings.detach().clone().requires_grad_(True)
    output = model.forward_group(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        candidate,
        batch.previous_actuator_flow,
        prepared,
        batch.elapsed_seconds,
        source_kind=batch.source_kind,
    )
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0].detach()
    changed = (candidate.detach() - batch.reference_settings[:, None]).abs() > 1e-8
    changed_gradient = gradient[changed]
    return {
        "finite_fraction": int(torch.isfinite(gradient).sum().cpu()) / gradient.numel(),
        "nonzero_fraction": int(gradient.ne(0.0).sum().cpu()) / gradient.numel(),
        "changed_action_nonzero_fraction": (
            int(changed_gradient.ne(0.0).sum().cpu()) / changed_gradient.numel()
            if changed_gradient.numel()
            else 0.0
        ),
        "max_abs": float(gradient.abs().max().cpu()),
        "l2_norm": float(torch.linalg.vector_norm(gradient).cpu()),
    }


def _trajectory_diagnostic(
    model: DifferentiableCounterfactualResponseModelV41,
    grouped_pairs: dict[str, list[Any]],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    by_source: dict[str, dict[str, list[float]]] = {}
    negative_count = 0
    flooding_count = 0
    for pairs in grouped_pairs.values():
        batch = stack_response_group_v41(pairs, device)
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind=batch.source_kind,
            )
        source = by_source.setdefault(
            batch.source_kind,
            {
                "delta_state_squared_error": [],
                "delta_flow_squared_error": [],
                "delta_flood_squared_error": [],
                "direct_tfv_abs_error": [],
                "trajectory_tfv_abs_error": [],
            },
        )
        state_error = output.delta_states_physical - batch.true_delta_states_physical
        flow_error = output.delta_flows_physical - batch.true_delta_flows_physical
        source["delta_state_squared_error"].append(float(state_error.square().mean().cpu()))
        source["delta_flow_squared_error"].append(float(flow_error.square().mean().cpu()))
        source["delta_flood_squared_error"].append(
            float(state_error[..., 2].square().mean().cpu())
        )
        source["direct_tfv_abs_error"].append(
            float((output.direct_delta_tfv_m3 - batch.true_delta_tfv_m3).abs().mean().cpu())
        )
        source["trajectory_tfv_abs_error"].append(
            float(
                (output.trajectory_delta_tfv_m3 - batch.true_delta_tfv_m3)
                .abs()
                .mean()
                .cpu()
            )
        )
        flood = output.candidate_states_physical[..., 2]
        negative_count += int((flood < 0.0).sum().cpu())
        flooding_count += flood.numel()
    return {
        source: {
            "delta_state_rmse_physical": float(
                np.sqrt(np.mean(values["delta_state_squared_error"]))
            ),
            "delta_flow_rmse_m3s": float(
                np.sqrt(np.mean(values["delta_flow_squared_error"]))
            ),
            "delta_flood_rate_rmse_m3s": float(
                np.sqrt(np.mean(values["delta_flood_squared_error"]))
            ),
            "direct_delta_tfv_mae_m3": float(np.mean(values["direct_tfv_abs_error"])),
            "trajectory_delta_tfv_mae_m3": float(
                np.mean(values["trajectory_tfv_abs_error"])
            ),
        }
        for source, values in by_source.items()
    } | {
        "physical_flooding_negative_fraction": float(
            negative_count / max(1, flooding_count)
        )
    }


def _mechanism_gate(metric: dict[str, Any], gradient: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "spread_same_order": 0.1 <= float(metric["spread_ratio"]) <= 10.0,
        "rank_material": float(metric["rank"]) >= 0.5,
        "sign_majority": float(metric["sign"]) > 0.5,
        "top1": bool(metric["top1"]),
        "gradient_finite": float(gradient["finite_fraction"]) == 1.0,
        "changed_action_gradient_nonzero": float(
            gradient["changed_action_nonzero_fraction"]
        )
        > 0.0,
    }
    return {
        "criteria_kind": "user_requested_tiny_mechanism_progression_not_formal_acceptance",
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _effect_records_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"candidate_count": 0}
    truth = np.asarray([record["truth"] for record in records], dtype=np.float64)
    predicted = np.asarray([record["predicted"] for record in records], dtype=np.float64)
    if len(records) >= 2:
        metrics = group_metrics_v41(
            predicted=predicted,
            truth=truth,
            group="diagnostic",
            source_kind=str(records[0]["source_kind"]),
        )
        rank = metrics["rank"]
        spread_ratio = metrics["spread_ratio"]
    else:
        rank = float("nan")
        spread_ratio = float("nan")
    meaningful = np.abs(truth) >= 1.0
    return {
        "candidate_count": len(records),
        "spread_ratio": spread_ratio,
        "rank": rank,
        "sign": (
            float(np.mean(np.sign(predicted[meaningful]) == np.sign(truth[meaningful])))
            if meaningful.any()
            else float("nan")
        ),
        "mean_abs_response_ratio": float(
            np.mean(np.abs(predicted)) / max(np.mean(np.abs(truth)), 1e-12)
        ),
        "mae_m3": float(np.mean(np.abs(predicted - truth))),
    }


def _magnitude_strata(
    contributions: list[dict[str, Any]], scales: CounterfactualDeltaScalesV41
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("D2", "D3"):
        quantiles = scales.by_source[source].tfv_abs_quantiles_m3
        q33, q67 = quantiles["q33"], quantiles["q67"]
        records = []
        for row in contributions:
            if row["source_kind"] != source:
                continue
            magnitude = abs(float(row["true_delta_tfv_m3"]))
            stratum = "small" if magnitude <= q33 else "medium" if magnitude <= q67 else "large"
            records.append(
                {
                    "source_kind": source,
                    "stratum": stratum,
                    "truth": float(row["true_delta_tfv_m3"]),
                    "predicted": float(row["predicted_final_delta_tfv_m3"]),
                }
            )
        result[source] = {
            "train_only_boundaries_m3": {"q33": q33, "q67": q67},
            "strata": {
                stratum: _effect_records_metrics(
                    [record for record in records if record["stratum"] == stratum]
                )
                for stratum in ("small", "medium", "large")
            },
        }
    return result


def _actuator_type(graph: Any, actuator_index: int) -> str:
    names = list(graph.actuator_physics_feature_names)
    for feature, label in (
        ("is_pump", "pump"),
        ("is_orifice", "orifice"),
        ("is_weir", "weir"),
        ("is_outlet", "outlet"),
    ):
        if feature in names and graph.actuator_physics[actuator_index, names.index(feature)] > 0.5:
            return label
    return "unknown"


def _d2_actuator_coverage(
    grouped_pairs: dict[str, list[Any]],
    contributions: list[dict[str, Any]],
    graph: Any,
) -> dict[str, Any]:
    predicted = {
        (row["group"], int(row["candidate_index"])): row
        for row in contributions
        if row["source_kind"] == "D2"
    }
    records: list[dict[str, Any]] = []
    invalid_active_counts: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        if not group.startswith("D2::"):
            continue
        for candidate_index, pair in enumerate(pairs):
            delta = pair.candidate["settings"] - pair.reference["settings"]
            active = np.flatnonzero(np.any(np.abs(delta) > 1e-8, axis=0))
            if active.size != 1:
                invalid_active_counts.append(
                    {"group": group, "candidate_index": candidate_index, "active_count": int(active.size)}
                )
                continue
            actuator_index = int(active[0])
            values = predicted[(group, candidate_index)]
            records.append(
                {
                    "source_kind": "D2",
                    "group": group,
                    "actuator_index": actuator_index,
                    "actuator_id": str(graph.actuator_ids[actuator_index]),
                    "actuator_type": _actuator_type(graph, actuator_index),
                    "truth": float(values["true_delta_tfv_m3"]),
                    "predicted": float(values["predicted_final_delta_tfv_m3"]),
                }
            )
    return {
        "invalid_single_actuator_candidates": invalid_active_counts,
        "by_type": {
            actuator_type: _effect_records_metrics(
                [record for record in records if record["actuator_type"] == actuator_type]
            )
            for actuator_type in ("pump", "orifice", "weir", "outlet", "unknown")
        },
        "by_identity": {
            actuator_id: _effect_records_metrics(
                [record for record in records if record["actuator_id"] == actuator_id]
            )
            for actuator_id in sorted({record["actuator_id"] for record in records})
        },
        "covered_actuator_count": len({record["actuator_id"] for record in records}),
        "candidate_count": len(records),
    }


def _simultaneous_action_diagnostic(
    model: DifferentiableCounterfactualResponseModelV41,
    pairs: list[Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch = stack_response_group_v41(pairs, device)
    results: dict[str, Any] = {}
    for count in (1, 5, 10, 20):
        candidate = batch.reference_settings[:, None].detach().clone()
        selected = candidate[..., :count]
        candidate[..., :count] = torch.where(selected <= 0.5, selected + 0.1, selected - 0.1)
        candidate.requires_grad_(True)
        output = model.forward_group(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            candidate,
            batch.previous_actuator_flow,
            prepared,
            batch.elapsed_seconds,
            source_kind="D3",
        )
        objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
        gradient = torch.autograd.grad(objective, candidate)[0].detach()
        changed = gradient[..., :count]
        results[str(count)] = {
            "finite": bool(torch.isfinite(gradient).all().cpu()),
            "all_changed_actuators_nonzero": bool(
                changed.abs().sum(dim=(0, 1, 2)).gt(0.0).all().cpu()
            ),
            "gradient_l2_norm": float(torch.linalg.vector_norm(changed).cpu()),
            "gradient_max_abs": float(changed.abs().max().cpu()),
            "representation_test_only": True,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_LAYOUT))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    study = Path(args.study_root).resolve()
    result_root = study / "step2_response_calibration_v41"
    v4 = study / "step2_control_response_v4"
    cache_dir = study / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
    full_manifest = study / "step2_counterfactual_formal_v1" / "02_shards" / "train" / "manifest.json"
    scale_path = result_root / "00_scales" / "counterfactual_delta_scales_train18.json"
    graph_path = study / "formal_assets" / "graph_schema.npz"
    v3_checkpoint = (
        study / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
    )
    old_scale_path = (
        study / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
    )
    tiny_groups = _read_groups(v4 / "02_tiny_overfit" / "02_tiny_overfit.groups.csv")
    micro_groups = _read_groups(v4 / "03_12_group_micro" / "03_12_group_micro.groups.csv")
    d2_tiny = [group for group in tiny_groups if group.startswith("D2::")]
    d3_tiny = [group for group in tiny_groups if group.startswith("D3::")]
    if len(d2_tiny) != 1 or len(d3_tiny) != 1:
        raise RuntimeError("frozen V4 tiny cohort must have exactly one D2 and one D3 group")
    if len(micro_groups) != 12 or not set(tiny_groups).issubset(micro_groups):
        raise RuntimeError("frozen V4 12-group cohort is missing or changed")

    cache = load_step2_training_cache(cache_dir / "CACHE_MANIFEST.json")
    split = {
        str(value).lower()
        for item in cache["shards"]
        for value in item["arrays"]["scientific_split"].tolist()
    }
    fold = {
        str(value).lower()
        for item in cache["shards"]
        for value in item["arrays"]["development_fold"].tolist()
    }
    if split != {"development"} or fold != {"train"}:
        raise RuntimeError(f"micro cache is not development/train only: {split=} {fold=}")

    scales, reused = load_or_derive_train_only_scales_v41(full_manifest, scale_path)
    if not reused:
        raise RuntimeError("bounded training expected the already-built full-Train scale cache")
    graph = _load_graph(graph_path)
    normalization = build_full_train_normalization_from_checkpoint(v3_checkpoint, old_scale_path)
    if normalization.source_manifest_sha256 != scales.source_manifest_sha256:
        raise RuntimeError("normalization and V4.1 scale artifacts do not bind the same Train manifest")

    if args.stage == "tiny-d2":
        selected = d2_tiny
        parent = None
    elif args.stage == "tiny-d3":
        _require_gate(result_root / "02_tiny_d2" / "stage_result.json", "D2")
        selected = d3_tiny
        parent = result_root / "02_tiny_d2" / "v41_tiny_d2.pt"
    elif args.stage == "tiny-combined":
        _require_gate(result_root / "02_tiny_d2" / "stage_result.json", "D2")
        _require_gate(result_root / "03_tiny_d3" / "stage_result.json", "D3")
        selected = tiny_groups
        parent = result_root / "03_tiny_d3" / "v41_tiny_d3.pt"
    else:
        _require_gate(result_root / "04_tiny_combined" / "stage_result.json", "D2")
        _require_gate(result_root / "04_tiny_combined" / "stage_result.json", "D3")
        selected = micro_groups
        parent = result_root / "04_tiny_combined" / "v41_tiny_combined.pt"

    grouped_pairs = load_train_groups(cache_dir, normalization, selected)
    model = _model(graph, normalization, scales, hidden_dim=args.hidden_dim)
    if parent is not None:
        _load_parent(model, parent)
    target_device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model.to(target_device).float()
    pretrain_prepared = prepare_graph_v41(model, graph, normalization, target_device)
    pretrain_metrics, _ = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=pretrain_prepared,
        device=target_device,
    )
    stage_dir_name, checkpoint_name = STAGE_LAYOUT[args.stage]
    stage_dir = result_root / stage_dir_name
    checkpoint = stage_dir / checkpoint_name
    training = train_response_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        normalization=normalization,
        scales=scales,
        graph=graph,
        out_path=checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=20260811,
        device=args.device,
    )
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    metrics, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=prepared,
        device=target_device,
    )
    gradients = {
        group: _gradient_diagnostic(model, pairs, prepared, target_device)
        for group, pairs in sorted(grouped_pairs.items())
    }
    gates = {
        source: _mechanism_gate(
            next(metric for metric in metrics if metric["source_kind"] == source),
            gradients[next(group for group in gradients if group.startswith(source + "::"))],
        )
        for source in sorted({metric["source_kind"] for metric in metrics})
        if sum(metric["source_kind"] == source for metric in metrics) == 1
    }
    if args.stage == "micro":
        gates = {
            source: {
                "criteria_kind": "diagnostic_only_no_formal_acceptance",
                "verdict": "NOT_A_GATE",
            }
            for source in ("D2", "D3")
        }
    simultaneous_group = next(
        (group for group in sorted(grouped_pairs) if group.startswith("D2::")),
        min(grouped_pairs),
    )
    result = {
        "contract": "STEP2_RESPONSE_CALIBRATION_V41_STAGE_RESULT",
        "stage": args.stage,
        "boundary": {
            "scientific_split": sorted(split),
            "development_fold": sorted(fold),
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_step2_run": False,
            "closed_loop_run": False,
            "precision": "fp32",
        },
        "repo": str(repo),
        "scale_manifest_sha256": scales.source_manifest_sha256,
        "selected_groups": selected,
        "parent_checkpoint": str(parent) if parent else None,
        "pretrain_group_metrics": pretrain_metrics,
        "training": training,
        "group_metrics": metrics,
        "candidate_contributions": contributions,
        "gradient_diagnostics": gradients,
        "trajectory_diagnostics": _trajectory_diagnostic(
            model, grouped_pairs, prepared, target_device
        ),
        "magnitude_strata": _magnitude_strata(contributions, scales),
        "d2_actuator_coverage": _d2_actuator_coverage(
            grouped_pairs, contributions, graph
        ),
        "simultaneous_action_diagnostics": _simultaneous_action_diagnostic(
            model,
            grouped_pairs[simultaneous_group],
            prepared,
            target_device,
        ),
        "mechanism_gates": gates,
    }
    _write_json(stage_dir / "stage_result.json", result)
    print(json.dumps({"stage": args.stage, "metrics": metrics, "gates": gates}, indent=2))
    if args.stage != "micro" and any(gate["verdict"] != "PASS" for gate in gates.values()):
        print("FAIL_CLOSED: do not run the next stage", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
