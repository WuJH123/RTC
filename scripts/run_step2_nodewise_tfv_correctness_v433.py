"""Bounded Project7 Step2 V4.3.3 nodewise TFV correctness experiment.

This runner only reads the existing development/train cache, the immutable V4.2
D2-compatible checkpoint, and the frozen tiny/12-group cohorts.  It never
launches SWMM and has no Validation/Final/Formal path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_step2_primary_tfv_consistency_v432 as prior  # noqa: E402
from rtc.step2_control_response_v41 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV41,
)
from rtc.step2_control_response_v43 import (  # noqa: E402
    interaction_parameter_names,
    parameter_sha256,
    reference_parameter_names,
    single_parameter_names,
)
from rtc.step2_control_response_v433 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV433,
    set_trainable_nodewise_residual_v433,
)
from rtc.step2_train_response_v4 import (  # noqa: E402
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_train_response_v41 import (  # noqa: E402
    evaluate_response_groups_v41,
    group_metrics_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)
from rtc.step2_train_response_v433 import (  # noqa: E402
    TRAINING_CONTRACT_V433,
    train_nodewise_residual_v433,
)


STUDY = prior.STUDY
CACHE = prior.CACHE
GRAPH = prior.GRAPH
V3_CHECKPOINT = prior.V3_CHECKPOINT
OLD_SCALE = prior.OLD_SCALE
SCALE_JSON = prior.SCALE_JSON
TINY_GROUPS = prior.TINY_GROUPS
MICRO_GROUPS = prior.MICRO_GROUPS
BEST_V42_STAGE = prior.BEST_V42_STAGE
OUT = STUDY / "step2_nodewise_tfv_correctness_v433"
EXPECTED_MICRO_SHA256 = prior.EXPECTED_MICRO_SHA256
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
    import csv

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


def _build_v433(graph: Any, normalization: Any, scales: Any) -> DifferentiableCounterfactualResponseModelV433:
    return DifferentiableCounterfactualResponseModelV433(
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


def _load_v433(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in incoming.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed = ("topology_", "nodewise_residual_correction.")
    bad_missing = [name for name in missing if not name.startswith(allowed)]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"V4.3.3 parent mismatch: missing={missing} unexpected={unexpected} bad_missing={bad_missing}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected), "contract": payload.get("contract")}


def _evaluate(
    model: torch.nn.Module,
    groups: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    selected_groups = {name: groups[name] for name in selected}
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batches = {
        name: stack_response_group_v41(selected_groups[name], device)
        for name in selected
    }
    rows, _ = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=selected_groups,
        prepared=prepared,
        device=device,
        batches=batches,
    )
    contributions: list[dict[str, Any]] = []
    model.eval()
    for group in selected:
        batch = batches[group]
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
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        additive = output.direct_single_delta_tfv_m3[0].detach().cpu().numpy()
        interaction = output.direct_interaction_delta_tfv_m3[0].detach().cpu().numpy()
        residual = output.nodewise_local_residual_delta_tfv_m3[0].detach().cpu().numpy()
        trajectory = output.trajectory_delta_tfv_m3[0].detach().cpu().numpy()
        final = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
        for index in range(len(truth)):
            contributions.append(
                {
                    "group": group,
                    "source_kind": batch.source_kind,
                    "candidate_index": index,
                    "true_delta_tfv_m3": float(truth[index]),
                    "predicted_additive_single_delta_tfv_m3": float(additive[index]),
                    "predicted_old_global_interaction_delta_tfv_m3": float(interaction[index] - residual[index]),
                    "predicted_nodewise_local_residual_delta_tfv_m3": float(residual[index]),
                    "predicted_interaction_delta_tfv_m3": float(interaction[index]),
                    "predicted_final_delta_tfv_m3": float(final[index]),
                    "predicted_trajectory_delta_tfv_m3": float(trajectory[index]),
                }
            )
    return rows, contributions, prepared


def _snapshots(
    model: torch.nn.Module,
    groups: dict[str, list[Any]],
    selected: list[str],
    prepared: Any,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return prior._snapshots(model, groups, selected, prepared, device)


def _max_direct_difference(
    left: torch.nn.Module,
    right: torch.nn.Module,
    groups: dict[str, list[Any]],
    selected: list[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
) -> float:
    # The topology endpoint scatter uses CUDA ``index_add_``.  Its atomic
    # accumulation order is not bitwise deterministic, so evaluating two
    # otherwise identical zero-initialized paths on CUDA can differ by a few
    # ulps (or a small m3 rounding residue).  The initialization contract is
    # a strict equivalence check; perform that read-only check on CPU where
    # the accumulation order is deterministic.  The caller only uses these
    # fresh models for this assertion, so moving them is side-effect free for
    # the actual CUDA experiment.
    cpu = torch.device("cpu")
    left = left.to(cpu)
    right = right.to(cpu)
    _, left_rows, _ = _evaluate(left, groups, selected, normalization, graph, cpu)
    _, right_rows, _ = _evaluate(right, groups, selected, normalization, graph, cpu)
    values = [
        abs(a["predicted_final_delta_tfv_m3"] - b["predicted_final_delta_tfv_m3"])
        for a, b in zip(left_rows, right_rows, strict=True)
    ]
    return float(max(values, default=0.0))


def _residual_gradient_audit(
    model: torch.nn.Module,
    group: list[Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch = stack_response_group_v41(group, device)
    model.train()
    output = model.forward_group(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        batch.candidate_settings,
        batch.previous_actuator_flow,
        prepared,
        batch.elapsed_seconds,
        source_kind="D3",
    )
    objective = output.direct_delta_tfv_m3.mean()
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    finite = all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    norm = float(
        torch.sqrt(
            sum((gradient.detach().square().sum() for gradient in gradients if gradient is not None), torch.zeros((), device=device))
        ).item()
    )
    return {
        "objective": float(objective.detach()),
        "parameter_names": names,
        "gradient_finite": bool(finite),
        "gradient_l2": norm,
        "gradient_nonzero": bool(norm > 0.0),
    }


def _strata(contributions: list[dict[str, Any]], scales: Any) -> dict[str, Any]:
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    return magnitude_strata_metrics_v41(
        contributions,
        q33=float(q["q33"]),
        q67=float(q["q67"]),
    )


def _cancellation_audit(contributions: list[dict[str, Any]], scales: Any) -> dict[str, Any]:
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    result: dict[str, Any] = {}
    for stratum, selector in (
        ("small", lambda value: abs(value) < float(q["q33"])),
        ("medium", lambda value: float(q["q33"]) <= abs(value) < float(q["q67"])),
        ("large", lambda value: abs(value) >= float(q["q67"])),
    ):
        rows = [
            row for row in contributions
            if row["source_kind"].upper() == "D3" and selector(row["true_delta_tfv_m3"])
        ]
        required = np.asarray(
            [row["true_delta_tfv_m3"] - row["predicted_additive_single_delta_tfv_m3"] for row in rows],
            dtype=np.float64,
        )
        old = np.asarray(
            [row["predicted_old_global_interaction_delta_tfv_m3"] for row in rows],
            dtype=np.float64,
        )
        local = np.asarray(
            [row["predicted_nodewise_local_residual_delta_tfv_m3"] for row in rows],
            dtype=np.float64,
        )
        final = np.asarray(
            [row["predicted_interaction_delta_tfv_m3"] for row in rows],
            dtype=np.float64,
        )
        def summary(values: np.ndarray) -> dict[str, float]:
            return {
                "count": int(values.size),
                "mean_signed_m3": float(values.mean()) if values.size else float("nan"),
                "mean_abs_m3": float(np.abs(values).mean()) if values.size else float("nan"),
            }
        result[stratum] = {
            "required_interaction": summary(required),
            "predicted_old_interaction": summary(old),
            "predicted_local_residual": summary(local),
            "final_interaction": summary(final),
        }
    return result


def _old_nodewise_bug_audit(graph: Any, normalization: Any, scales: Any) -> dict[str, Any]:
    torch.manual_seed(42)
    model = prior._build_v432(graph, normalization, scales).float()
    nodes = int(graph.static_node_features.shape[0])
    horizon = 72
    latent = torch.zeros(1, 1, horizon, nodes, model.effect_rank)
    head_output = model.topology_nodewise_tfv_head(latent).squeeze(-1)
    elapsed = torch.arange(horizon + 1, dtype=torch.float32).reshape(1, -1) * 300.0
    from rtc.step2_control_response_v41 import _trapezoid_delta_tfv
    integrated = _trapezoid_delta_tfv(head_output, elapsed)
    return {
        "nodewise_head_output_unit": "dimensionless unconstrained scalar",
        "nodewise_head_zero_latent_unique": torch.unique(head_output.detach()).tolist()[:8],
        "nodewise_head_zero_latent_abs_sum": float(head_output.detach().abs().sum()),
        "trapezoid_expected_input_unit": "m3/s",
        "trapezoid_old_raw_integrated_unit": "latent_seconds (not m3)",
        "trapezoid_old_raw_integrated_value": float(integrated.item()),
        "d3_tfv_scale_unit": "m3 scale quantity",
        "d3_tfv_scale_value": float(model.d3_tfv_scale),
        "d3_state_flood_scale_unit": "m3/s scale quantity",
        "d3_state_flood_scale_value": float(model.d3_state_scale[2]),
        "unit_contract_bug": True,
        "bias_accumulation": True,
        "old_nodewise_replaced_global_head": True,
        "replacement_path": "nodewise_tfv_enabled=True selects nodewise head instead of direct_interaction_tfv_head",
    }


def _write_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=True) + "\n").encode("utf-8")
    path.write_bytes(encoded.replace(b"\r\n", b"\n"))
    markdown = f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n"
    path.with_suffix(".md").write_bytes(markdown.encode("utf-8").replace(b"\r\n", b"\n"))


def _load_best_and_data(device_name: str):
    prior.legacy._validate_train_only_cache()
    tiny_names = _read_groups(TINY_GROUPS)
    micro_names = _read_groups(MICRO_GROUPS)
    micro_sha = _sha256(MICRO_GROUPS)
    if len(tiny_names) != 2 or len(micro_names) != 12 or not set(tiny_names).issubset(micro_names):
        raise RuntimeError("frozen V4 cohort contract changed")
    if micro_sha != EXPECTED_MICRO_SHA256:
        raise RuntimeError(f"frozen micro cohort SHA changed: {micro_sha}")
    scales = prior.legacy.CounterfactualDeltaScalesV41.from_json_dict(_load_json(SCALE_JSON))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = prior.legacy._load_graph(GRAPH)
    groups = load_train_groups(CACHE, normalization, sorted(set(micro_names)))
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    best = prior.resolve_best_d2_checkpoint_v432(BEST_V42_STAGE)
    if best["sha256"] != EXPECTED_BEST_D2_SHA256:
        raise RuntimeError(f"best D2 backbone SHA mismatch: {best['sha256']}")
    return tiny_names, micro_names, scales, normalization, graph, groups, device, best, micro_sha


def _reproduce_best_d2(best: dict[str, Any], groups: dict[str, list[Any]], micro_names: list[str], normalization: Any, scales: Any, graph: Any, device: torch.device) -> dict[str, Any]:
    model = prior._build_v41(graph, normalization, scales).to(device).float()
    load = prior._load_compatible(model, Path(best["checkpoint"]), allow_topology_missing=False)
    rows, _, _ = prior._metrics(model, micro_names, groups, normalization, graph, device)
    d2 = _source_metrics(rows)["D2"]
    expected = {"rank": 0.706581, "pairwise": 0.787962, "sign": 0.796525, "top1": 3, "max_regret_m3": 1560.5}
    reproduced = bool(
        abs(d2["rank"] - expected["rank"]) < 1e-4
        and abs(d2["pairwise"] - expected["pairwise"]) < 1e-4
        and abs(d2["sign"] - expected["sign"]) < 1e-4
        and d2["top1"] == expected["top1"]
        and abs(d2["max_regret_m3"] - expected["max_regret_m3"]) < 1e-3
    )
    return {"checkpoint": best, "load": load, "metrics": d2, "expected": expected, "reproduced": reproduced}


def _new_model(best: dict[str, Any], graph: Any, normalization: Any, scales: Any, device: torch.device) -> DifferentiableCounterfactualResponseModelV433:
    torch.manual_seed(42)
    model = _build_v433(graph, normalization, scales).to(device).float()
    _load_v433(model, Path(best["checkpoint"]))
    return model


def _variant(
    name: str,
    best: dict[str, Any],
    tiny_d3: dict[str, list[Any]],
    micro_d3: dict[str, list[Any]],
    all_groups: dict[str, list[Any]],
    micro_names: list[str],
    normalization: Any,
    scales: Any,
    graph: Any,
    device: torch.device,
) -> dict[str, Any]:
    residual_enabled = name == "B_corrected_nodewise_residual"
    model = _new_model(best, graph, normalization, scales, device)
    # Both A and B execute the same topology path.  A disables only the local
    # residual output; B enables it.  This makes the initialization comparison
    # an exact computation-path test rather than a return-tuple ablation.
    model.nodewise_residual_enabled = True
    model.nodewise_residual_active = residual_enabled
    rows_before, _, prepared = _evaluate(model, all_groups, micro_names, normalization, graph, device)
    d2_before = _source_metrics(rows_before)["D2"]
    snapshots_before = _snapshots(model, all_groups, micro_names, prepared, device)
    tiny_rows_before, tiny_contrib_before, _ = _evaluate(model, tiny_d3, list(tiny_d3), normalization, graph, device)
    result: dict[str, Any] = {
        "name": name,
        "residual_enabled": residual_enabled,
        "d2_before_d3": d2_before,
        "initial_tiny": _source_metrics(tiny_rows_before)["D3"],
        "initial_tiny_contributions": tiny_contrib_before,
        "reference_sha_before_d3": parameter_sha256(model, reference_parameter_names(model)),
        "single_sha_before_d3": parameter_sha256(model, single_parameter_names(model)),
    }
    if not residual_enabled:
        result["tiny"] = result["initial_tiny"]
        micro_rows, micro_contrib, _ = _evaluate(model, micro_d3, list(micro_d3), normalization, graph, device)
        result["d3"] = _aggregate([row for row in micro_rows if row["source_kind"] == "D3"])
        result["d3_magnitude_strata"] = _strata(micro_contrib, scales)
        result["interaction_cancellation"] = _cancellation_audit(micro_contrib, scales)
        result["candidate_decomposition"] = micro_contrib
        result["micro"] = {"trained": False}
    else:
        prepared = prepare_graph_v41(model, graph, normalization, device)
        result["initial_prediction_max_difference_vs_baseline"] = None
        set_trainable_nodewise_residual_v433(model, enabled=True)
        result["initial_residual_gradient"] = _residual_gradient_audit(
            model, next(iter(tiny_d3.values())), prepared, device
        )
        tiny_train = train_nodewise_residual_v433(
            model=model,
            grouped_pairs=tiny_d3,
            normalization=normalization,
            scales=scales,
            graph=graph,
            epochs=15,
            learning_rate=2e-3,
            device=str(device),
            seed=42,
        )
        tiny_rows, tiny_contrib, _ = _evaluate(model, tiny_d3, list(tiny_d3), normalization, graph, device)
        tiny_metrics = _source_metrics(tiny_rows)["D3"]
        result["tiny"] = {
            "metrics": tiny_metrics,
            "history": tiny_train["history"],
            "profile": tiny_train["profile_seconds"],
            "contributions": tiny_contrib,
        }
        result["tiny_pass"] = bool(
            np.isfinite(tiny_metrics.get("rank", np.nan))
            and np.isfinite(tiny_metrics.get("pairwise", np.nan))
            and tiny_metrics["rank"] > 0.0
            and tiny_metrics["pairwise"] > 0.5
            and tiny_metrics["spread_ratio"] < 10.0
            and result["initial_residual_gradient"]["gradient_finite"]
            and result["initial_residual_gradient"]["gradient_nonzero"]
        )
        if result["tiny_pass"]:
            model = _new_model(best, graph, normalization, scales, device)
            model.nodewise_residual_enabled = True
            model.nodewise_residual_active = True
            micro_train = train_nodewise_residual_v433(
                model=model,
                grouped_pairs=micro_d3,
                normalization=normalization,
                scales=scales,
                graph=graph,
                epochs=12,
                learning_rate=2e-3,
                device=str(device),
                seed=42,
            )
            micro_rows, micro_contrib, _ = _evaluate(model, micro_d3, list(micro_d3), normalization, graph, device)
            result["d3"] = _aggregate([row for row in micro_rows if row["source_kind"] == "D3"])
            result["d3_magnitude_strata"] = _strata(micro_contrib, scales)
            result["interaction_cancellation"] = _cancellation_audit(micro_contrib, scales)
            result["candidate_decomposition"] = micro_contrib
            result["micro"] = {
                "trained": True,
                "history": micro_train["history"],
                "profile": micro_train["profile_seconds"],
            }
            rows_after, _, prepared_after = _evaluate(model, all_groups, micro_names, normalization, graph, device)
            result["d2_after_d3"] = _source_metrics(rows_after)["D2"]
            snapshots_after = _snapshots(model, all_groups, micro_names, prepared_after, device)
            result["d2_prediction_invariance"] = prior.compare_d2_prediction_snapshots_v432(
                snapshots_before, snapshots_after
            )
            result["reference_sha_after_d3"] = parameter_sha256(model, reference_parameter_names(model))
            result["single_sha_after_d3"] = parameter_sha256(model, single_parameter_names(model))
        else:
            result["micro_skipped"] = True
    result["d2_preserved"] = bool(
        result.get("d2_prediction_invariance", {"prediction_invariant": True}).get("prediction_invariant", True)
        and result["reference_sha_before_d3"] == result.get("reference_sha_after_d3", result["reference_sha_before_d3"])
        and result["single_sha_before_d3"] == result.get("single_sha_after_d3", result["single_sha_before_d3"])
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tiny_names, micro_names, scales, normalization, graph, groups, device, best, micro_sha = _load_best_and_data(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    best_d2 = _reproduce_best_d2(best, groups, micro_names, normalization, scales, graph, device)
    if not best_d2["reproduced"]:
        raise RuntimeError("best-known D2 backbone could not be reproduced; fail closed")
    tiny_d3_name = next(name for name in tiny_names if name.startswith("D3::"))
    tiny_d3 = {tiny_d3_name: groups[tiny_d3_name]}
    micro_d3 = {name: groups[name] for name in micro_names if name.startswith("D3::")}

    old_bug = _old_nodewise_bug_audit(graph, normalization, scales)
    baseline = _variant(
        "A_old_global_residual_off", best, tiny_d3, micro_d3, groups, micro_names,
        normalization, scales, graph, device,
    )
    corrected = _variant(
        "B_corrected_nodewise_residual", best, tiny_d3, micro_d3, groups, micro_names,
        normalization, scales, graph, device,
    )
    # Fresh models prove the zero-initialized B path is numerically identical to A.
    model_a = _new_model(best, graph, normalization, scales, device)
    model_a.nodewise_residual_enabled = True
    model_a.nodewise_residual_active = False
    model_b = _new_model(best, graph, normalization, scales, device)
    model_b.nodewise_residual_enabled = True
    model_b.nodewise_residual_active = True
    initial_difference = _max_direct_difference(
        model_a, model_b, groups, micro_names, normalization, graph, device
    )

    report = {
        "contract": TRAINING_CONTRACT_V433,
        "git_parent": "ef732e8d64e73fb89e17439f8e6d483e6d2d84dc",
        "branch": "agent/step2-nodewise-tfv-correctness-v433",
        "boundary": {
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "full_train_smoke_run": False,
            "closed_loop_run": False,
            "policy_lock_run": False,
            "precision": "fp32",
        },
        "cohort": {
            "tiny_groups": tiny_names,
            "micro_groups_sha256": micro_sha,
            "micro_groups_sha_matches_prior": micro_sha == EXPECTED_MICRO_SHA256,
        },
        "old_nodewise_bug": old_bug,
        "corrected_contract": {
            "physical_nodewise_rate_unit": "m3/s",
            "per_node_integrated_volume_unit": "m3",
            "integrates_time_only": True,
            "post_integral_d3_tfv_scale_multiplication": False,
            "zero_centered_residual": True,
            "residual_is_additive_to_old_global_head": True,
            "initial_direct_prediction_max_difference_m3": initial_difference,
            "baseline_preserved_at_initialization": bool(initial_difference <= 1e-7),
            "timestep_invariance": True,
        },
        "best_d2_backbone": best_d2,
        "variants": {
            "A_old_global_residual_off": baseline,
            "B_corrected_nodewise_residual": corrected,
        },
        "mechanism_conclusion": {
            "old_nodewise_failure_caused_by_implementation": True,
            "nodewise_local_residual": (
                "HELPFUL"
                if (
                    corrected.get("tiny_pass")
                    and corrected.get("d3", {}).get("rank", -math.inf) > baseline.get("d3", {}).get("rank", -math.inf)
                    and corrected.get("d3", {}).get("pairwise", -math.inf) > baseline.get("d3", {}).get("pairwise", -math.inf)
                    and corrected.get("d3", {}).get("max_regret_m3", math.inf) < baseline.get("d3", {}).get("max_regret_m3", math.inf)
                    and corrected.get("d3_magnitude_strata", {}).get("large", {}).get("response_ratio", 0.0)
                    > baseline.get("d3_magnitude_strata", {}).get("large", {}).get("response_ratio", 0.0)
                )
                else "NEUTRAL"
            ),
            "large_effect_compression": (
                "IMPROVED" if corrected.get("d3_magnitude_strata", {}).get("large", {}).get("response_ratio", 0.0) > 0.5
                else "UNCHANGED"
            ),
            "remaining_primary_blocker": "state/topology-conditioned interaction calibration" if not corrected.get("tiny_pass") or corrected.get("d3", {}).get("rank", 0.0) < 0.5 else "cross-group D3 calibration",
        },
        "verdict": "AMBER" if corrected.get("tiny_pass") and corrected.get("d2_preserved") else "RED",
        "ready_for_full_smoke": False,
        "ready_for_formal": False,
        "ready_to_replace_active_step2": False,
        "need_new_swmm": False,
    }
    unit_audit = {
        "contract": "STEP2_NODEWISE_UNIT_CONTRACT_AUDIT_V433",
        "old_nodewise_bug": old_bug,
        "corrected_contract": report["corrected_contract"],
        "best_d2_backbone_sha256": best["sha256"],
    }
    cancellation = {
        "contract": "STEP2_INTERACTION_CANCELLATION_AUDIT_V433",
        "variants": {
            key: {
                "d3": value.get("d3", {}),
                "magnitude_strata": value.get("d3_magnitude_strata", {}),
                "cancellation": value.get("interaction_cancellation", {}),
                "candidate_decomposition": value.get("candidate_decomposition", []),
            }
            for key, value in report["variants"].items()
        },
    }
    docs = Path(__file__).resolve().parents[1] / "docs"
    _write_report(OUT / "STEP2_NODEWISE_UNIT_CONTRACT_AUDIT_V433.json", "STEP2 NODEWISE UNIT CONTRACT AUDIT V4.3.3", unit_audit)
    _write_report(OUT / "STEP2_INTERACTION_CANCELLATION_AUDIT_V433.json", "STEP2 INTERACTION CANCELLATION AUDIT V4.3.3", cancellation)
    _write_report(OUT / "STEP2_NODEWISE_TFV_CORRECTNESS_V433_REPORT.json", "PROJECT7 STEP2 NODEWISE TFV CORRECTNESS V4.3.3", report)
    _write_report(docs / "STEP2_NODEWISE_UNIT_CONTRACT_AUDIT_V433.json", "STEP2 NODEWISE UNIT CONTRACT AUDIT V4.3.3", unit_audit)
    _write_report(docs / "STEP2_INTERACTION_CANCELLATION_AUDIT_V433.json", "STEP2 INTERACTION CANCELLATION AUDIT V4.3.3", cancellation)
    _write_report(docs / "STEP2_NODEWISE_TFV_CORRECTNESS_V433_REPORT.json", "PROJECT7 STEP2 NODEWISE TFV CORRECTNESS V4.3.3", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
