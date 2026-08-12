"""Bounded Train-only V4.5 direct joint-action experiment.

The runner performs the superposition audit first, then fits fresh tiny and
12-group in-sample mechanism models from the immutable best D2 backbone.  It
never launches SWMM and has no Validation, Final, Formal, production, or smoke
path.
"""

from __future__ import annotations

import argparse
import copy
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

from rtc.production_cli import _load_graph  # noqa: E402
from rtc.step2_control_response_v45 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV45,
    direct_effect_parameter_names_v45,
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
from rtc.step2_train_response_v45 import train_direct_multi_actuator_v45  # noqa: E402


ROOT = Path(r"E:\RTC_sewer\Project7")
STUDY = ROOT / "study_v069"
V4 = STUDY / "step2_control_response_v4"
OUT = STUDY / "step2_direct_multiactuator_v45"
DOCS = ROOT / "repo" / "docs"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
MICRO_CACHE = STUDY / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
OLD_NORMALIZATION_SCALE_JSON = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
SCALE_JSON = STUDY / "step2_response_calibration_v41" / "00_scales" / "counterfactual_delta_scales_train18.json"
BEST_D2 = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "v42_12_group_micro.pt"
TINY_GROUPS = V4 / "02_tiny_overfit" / "02_tiny_overfit.groups.csv"
MICRO_GROUPS = V4 / "03_12_group_micro" / "03_12_group_micro.groups.csv"
EXPECTED_MICRO_SHA = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
EXPECTED_BEST_D2_SHA = "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"
OLD_V441_REPORT = DOCS / "STEP2_EDGE_PHYSICS_CORRECTNESS_V441_REPORT.json"
SUPERPOSITION_REPORT = DOCS / "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45.json"


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
        "max_regret_m3": float(np.max([row["regret_m3"] for row in rows])),
    }


def _source_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {source: _aggregate([row for row in rows if row["source_kind"].upper() == source]) for source in ("D2", "D3")}


def _build_model(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> DifferentiableCounterfactualResponseModelV45:
    return DifferentiableCounterfactualResponseModelV45(
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


def _load_best(model: torch.nn.Module) -> dict[str, Any]:
    sha = _sha256(BEST_D2)
    if sha != EXPECTED_BEST_D2_SHA:
        raise RuntimeError(f"best D2 backbone SHA mismatch: {sha}")
    payload = torch.load(BEST_D2, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    compatible = {name: value for name, value in incoming.items() if name in current and tuple(value.shape) == tuple(current[name].shape)}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed_missing = set(direct_effect_parameter_names_v45(model))
    bad_missing = [name for name in missing if name not in allowed_missing]
    if unexpected or bad_missing:
        raise RuntimeError(f"V4.5 parent mismatch: {missing=} {unexpected=} {bad_missing=}")
    return {"checkpoint": str(BEST_D2), "sha256": sha, "missing_direct_parameters": sorted(set(missing) & allowed_missing), "unexpected": list(unexpected)}


def _evaluate(model: torch.nn.Module, groups: dict[str, list[Any]], selected: list[str], normalization: Any, graph: Any, device: torch.device):
    selected_groups = {name: groups[name] for name in selected}
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batches = {name: stack_response_group_v41(selected_groups[name], device) for name in selected}
    rows, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=selected_groups,
        prepared=prepared,
        device=device,
        batches=batches,
    )
    return rows, contributions, prepared, batches


def _parameter_sha(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _reference_sha(model: torch.nn.Module) -> str:
    return _parameter_sha(model, ("node_static_encoder.", "actuator_static_encoder.", "actuator_identity.", "reference_"))


def _run_fresh(
    *,
    name: str,
    selected: list[str],
    groups: dict[str, list[Any]],
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    device_name: str,
    epochs: int,
) -> tuple[DifferentiableCounterfactualResponseModelV45, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(42)
    np.random.seed(42)
    model = _build_model(graph, normalization, scales)
    parent = _load_best(model)
    ref_before = _reference_sha(model)
    result = train_direct_multi_actuator_v45(
        model=model,
        grouped_pairs={group: groups[group] for group in selected},
        normalization=normalization,
        scales=scales,
        graph=graph,
        epochs=epochs,
        learning_rate=2e-3,
        device=device_name,
        seed=42,
    )
    ref_after = _reference_sha(model)
    result["name"] = name
    result["parent"] = parent
    result["reference_sha_before"] = ref_before
    result["reference_sha_after"] = ref_after
    result["reference_frozen"] = ref_before == ref_after
    result["selected_groups"] = list(selected)
    checkpoint = OUT / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"contract": result["contract"], "model_state_dict": model.state_dict(), "seed": 42, "precision": "fp32"}, checkpoint)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint)
    rows, contributions, _, _ = _evaluate(model, groups, selected, normalization, graph, torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu"))
    return model, result, rows, contributions


def _gradient_diagnostic(model: torch.nn.Module, group: list[Any], normalization: Any, graph: Any, device: torch.device) -> dict[str, Any]:
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batch = stack_response_group_v41(group, device)
    result: dict[str, Any] = {}
    for count in (5, 10, 20):
        candidate = batch.reference_settings[:, None].expand(-1, 1, -1, -1).clone()
        candidate[:, 0, :, :count] = candidate[:, 0, :, :count] + 0.2
        candidate.requires_grad_(True)
        output = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, candidate, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
        objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
        gradient = torch.autograd.grad(objective, candidate, retain_graph=False)[0]
        changed = (candidate.detach() - batch.reference_settings[:, None]).abs() > 1e-8
        changed_gradient = gradient[changed]
        result[f"{count}-device"] = {
            "finite_fraction": float(torch.isfinite(gradient).float().mean().detach().cpu()),
            "changed_nonzero_fraction": float(changed_gradient.ne(0.0).float().mean().detach().cpu()),
            "gradient_median_abs": float(changed_gradient.abs().median().detach().cpu()),
            "gradient_max_abs": float(changed_gradient.abs().max().detach().cpu()),
            "positive_fraction": float((changed_gradient > 0).float().mean().detach().cpu()),
            "negative_fraction": float((changed_gradient < 0).float().mean().detach().cpu()),
        }
    return result


def _candidate_order_jacobian(model: torch.nn.Module, group: list[Any], normalization: Any, graph: Any, device: torch.device) -> dict[str, Any]:
    prepared = prepare_graph_v41(model, graph, normalization, device)
    batch = stack_response_group_v41(group, device)
    candidate = batch.candidate_settings.detach().clone().requires_grad_(True)
    output = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, candidate, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind="D3")
    truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
    predicted = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
    metric = group_metrics_v41(predicted=predicted, truth=truth, group="jacobian", source_kind="D3")
    jacobian = []
    for index in range(candidate.shape[1]):
        grad = torch.autograd.grad(output.direct_delta_tfv_m3[0, index], candidate, retain_graph=index + 1 < candidate.shape[1])[0][0, index]
        jacobian.append({
            "candidate_index": index,
            "finite": bool(torch.isfinite(grad).all()),
            "nonzero_fraction": float(grad.ne(0.0).float().mean().detach().cpu()),
            "sign_vs_truth": float((torch.sign(grad[grad.ne(0.0)]).mean().detach().cpu())) if bool(grad.ne(0.0).any()) else 0.0,
        })
    return {"group_metric": metric, "jacobian": jacobian, "interpretation": "gradient sign is a local action-sensitivity diagnostic, not a SWMM finite-difference substitute"}


def _magnitude(contributions: list[dict[str, Any]], scales: CounterfactualDeltaScalesV41) -> dict[str, Any]:
    return magnitude_strata_metrics_v41(contributions, q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]), q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]))


def _comparison(v441_report: dict[str, Any], v45_contributions: list[dict[str, Any]], scales: CounterfactualDeltaScalesV41) -> dict[str, Any]:
    d3_v441 = [row for row in v441_report["micro"]["candidate_edge_contributions"] if row["source_kind"] == "D3"]
    v45 = {(row["group"], int(row["candidate_index"])): row for row in v45_contributions if row["source_kind"] == "D3"}
    variants = {"V4.3.3_additive_interaction": [], "V4.4.1_corrected_edge": [], "V4.5_direct": []}
    for row in d3_v441:
        key = (row["group"], int(row["candidate_index"]))
        direct = v45[key]
        variants["V4.3.3_additive_interaction"].append({"group": row["group"], "source_kind": "D3", "true_delta_tfv_m3": row["true_delta_tfv_m3"], "predicted_final_delta_tfv_m3": row["predicted_edge_off_final_delta_tfv_m3"]})
        variants["V4.4.1_corrected_edge"].append({"group": row["group"], "source_kind": "D3", "true_delta_tfv_m3": row["true_delta_tfv_m3"], "predicted_final_delta_tfv_m3": row["predicted_final_delta_tfv_m3"]})
        variants["V4.5_direct"].append({"group": direct["group"], "source_kind": "D3", "true_delta_tfv_m3": direct["true_delta_tfv_m3"], "predicted_final_delta_tfv_m3": direct["predicted_final_delta_tfv_m3"]})
    result = {}
    for name, rows in variants.items():
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(row["group"], []).append(row)
        metrics = [group_metrics_v41(predicted=np.asarray([r["predicted_final_delta_tfv_m3"] for r in group_rows]), truth=np.asarray([r["true_delta_tfv_m3"] for r in group_rows]), group=group, source_kind="D3") for group, group_rows in sorted(by_group.items())]
        result[name] = {"metrics": _aggregate(metrics), "magnitude": magnitude_strata_metrics_v41(rows, q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]), q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]))}
    return result


def run(device_name: str = "cuda") -> dict[str, Any]:
    if not SUPERPOSITION_REPORT.is_file():
        raise RuntimeError("superposition audit is missing; fail closed")
    superposition = json.loads(SUPERPOSITION_REPORT.read_text(encoding="utf-8"))
    if superposition.get("verdict") != "SUPERPOSITION INVALID":
        raise RuntimeError("superposition audit does not authorize V4.5; fail closed")
    graph = _load_graph(str(GRAPH))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_NORMALIZATION_SCALE_JSON)
    scales = CounterfactualDeltaScalesV41.from_json_dict(json.loads(SCALE_JSON.read_text(encoding="utf-8")))
    groups = load_train_groups(MICRO_CACHE, normalization)
    tiny_names = _read_groups(TINY_GROUPS)
    micro_names = _read_groups(MICRO_GROUPS)
    micro_sha = _sha256(MICRO_GROUPS)
    if micro_sha != EXPECTED_MICRO_SHA:
        raise RuntimeError(f"frozen micro cohort SHA changed: {micro_sha}")
    if len(tiny_names) != 2 or len(micro_names) != 12 or not set(tiny_names).issubset(micro_names):
        raise RuntimeError("frozen tiny/micro cohort contract changed")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    tiny_model, tiny_result, tiny_rows, tiny_contributions = _run_fresh(name="tiny_direct", selected=tiny_names, groups=groups, graph=graph, normalization=normalization, scales=scales, device_name=str(device), epochs=15)
    tiny_metrics = _source_metrics(tiny_rows)
    tiny_pass = bool(
        tiny_result["reference_frozen"]
        and tiny_metrics["D2"].get("rank", -math.inf) > 0.0
        and tiny_metrics["D3"].get("rank", -math.inf) > 0.0
        and tiny_metrics["D3"].get("pairwise", -math.inf) > 0.5
        and tiny_metrics["D3"].get("top1", 0) >= 1
    )
    micro_result: dict[str, Any] | None = None
    micro_rows: list[dict[str, Any]] = []
    micro_contributions: list[dict[str, Any]] = []
    micro_model: DifferentiableCounterfactualResponseModelV45 | None = None
    if tiny_pass:
        micro_model, micro_result, micro_rows, micro_contributions = _run_fresh(name="micro_direct", selected=micro_names, groups=groups, graph=graph, normalization=normalization, scales=scales, device_name=str(device), epochs=12)
    else:
        micro_result = {"skipped": True, "reason": "tiny gate failed"}
    v441_report = json.loads(OLD_V441_REPORT.read_text(encoding="utf-8"))
    comparison = _comparison(v441_report, micro_contributions, scales) if micro_contributions else {}
    diagnostic_model = micro_model if micro_model is not None else tiny_model
    diagnostic_names = micro_names if micro_model is not None else tiny_names
    diagnostic_group = groups[next(name for name in diagnostic_names if name.startswith("D3::"))]
    gradients = _gradient_diagnostic(diagnostic_model, diagnostic_group, normalization, graph, device)
    jacobian = _candidate_order_jacobian(diagnostic_model, diagnostic_group, normalization, graph, device)
    d3 = _aggregate([row for row in micro_rows if row["source_kind"] == "D3"]) if micro_rows else {}
    baseline_d3 = v441_report["micro"]["baseline_v433"]["D3"]
    corrected_d3 = v441_report["micro"]["corrected_v441"]["D3"]
    direct_d3_mag = _magnitude(micro_contributions, scales) if micro_contributions else {}
    direct_d2 = _aggregate([row for row in micro_rows if row["source_kind"] == "D2"]) if micro_rows else {}
    best_d2 = v441_report["best_d2_backbone"]["metrics"]["metrics"]
    direct_supported = bool(
        tiny_pass
        and d3.get("rank", -math.inf) > baseline_d3.get("rank", math.inf)
        and d3.get("pairwise", -math.inf) > baseline_d3.get("pairwise", math.inf)
        and d3.get("top1", 0) >= baseline_d3.get("top1", 0)
        and d3.get("mean_regret_m3", math.inf) < baseline_d3.get("mean_regret_m3", -math.inf)
        and direct_d3_mag.get("large", {}).get("response_ratio", 0.0) > 0.4213015
    )
    report = {
        "contract": "PROJECT7_STEP2_DIRECT_MULTI_ACTUATOR_V45",
        "git_parent": "37f57366e623f42bd809e34de9a9af5b761940dd",
        "branch": "agent/step2-direct-multiactuator-v45",
        "boundary": {"swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False, "validation_outcomes_accessed": False, "final_accessed": False, "formal_run": False, "full_train_smoke_run": False, "production_wiring_modified": False, "precision": "FP32"},
        "superposition_audit": {"report": str(SUPERPOSITION_REPORT), "verdict": superposition["verdict"], "authorized": bool(superposition.get("direct_v45_authorized")), "medium_large": superposition["medium_large_burden"], "overall_additive_candidate_metrics": superposition["overall_additive_candidate_metrics"]},
        "direct_model_contract": {"candidate_reference_shared_encoder": True, "effect_encoder": "effect_encoder_v45", "joint_context": "sum+mean+max+low-rank second-order before node scatter", "edge_v441_residual": "OFF", "reference_frozen": bool(tiny_result.get("reference_frozen", False) and (micro_result is None or micro_result.get("skipped", False) or micro_result.get("reference_frozen", False))), "structural_non_additivity": True, "causal_memory": "fixed 0.65", "hard_actuator_count_switch": False},
        "tiny": {"metrics": tiny_metrics, "passed": tiny_pass, "result": tiny_result, "baseline_v433": v441_report["tiny"]["baseline_v433"], "old_v44": v441_report["tiny"]["old_v44"], "corrected_v441": v441_report["tiny"]["corrected_v441"], "gradient_diagnostic": gradients, "candidate_order_jacobian": jacobian, "checkpoint_not_used_for_micro": True},
        "micro": {"in_sample_mechanism_micro": True, "cohort_sha256": micro_sha, "result": micro_result, "D2": direct_d2, "D3": d3, "magnitude": direct_d3_mag, "gradients": gradients if micro_model is not None else {}, "candidate_order_jacobian": jacobian if micro_model is not None else {}},
        "action_gradient": gradients,
        "comparison": comparison,
        "historical_best_d2": best_d2,
        "mechanism_conclusion": {"additive_d2_decomposition": "INVALID", "direct_joint_action_response": "SUPPORTED" if direct_supported else "INSUFFICIENT", "large_effect_compression": "IMPROVED" if direct_d3_mag.get("large", {}).get("response_ratio", 0.0) > 0.4213015 else ("NOT_ASSESSED_TINY_GATE_FAILED" if not micro_contributions else "UNCHANGED"), "primary_remaining_blocker": "NONE_WITHIN_BOUNDED_MICRO" if direct_supported else "DATA IDENTIFIABILITY AUDIT"},
        "verdict": "AMBER" if direct_supported else "RED",
        "ready_for_full_train_smoke": False,
        "ready_for_formal": False,
        "ready_to_replace_active_step2": False,
        "need_new_swmm": False,
        "next_bounded_action": "TRAIN-ONLY INTERNAL HOLDOUT" if direct_supported else "DATA IDENTIFIABILITY AUDIT",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    audit = {
        "contract": report["contract"],
        "audit_scope": "Train-only additive-vs-direct mechanism audit; micro is fail-closed when tiny gate fails",
        "boundary": report["boundary"],
        "superposition_audit": report["superposition_audit"],
        "direct_model_contract": report["direct_model_contract"],
        "tiny": report["tiny"],
        "micro": report["micro"],
        "action_gradient": report["action_gradient"],
        "historical_best_d2": report["historical_best_d2"],
        "mechanism_conclusion": report["mechanism_conclusion"],
        "verdict": report["verdict"],
    }
    for path in (OUT / "STEP2_DIRECT_MULTI_ACTUATOR_AUDIT_V45.json", DOCS / "STEP2_DIRECT_MULTI_ACTUATOR_AUDIT_V45.json"):
        path.write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    audit_markdown = "# STEP2 DIRECT MULTI-ACTUATOR V4.5 AUDIT\n\n```json\n" + json.dumps(audit, indent=2, allow_nan=True) + "\n```\n"
    for path in (OUT / "STEP2_DIRECT_MULTI_ACTUATOR_AUDIT_V45.md", DOCS / "STEP2_DIRECT_MULTI_ACTUATOR_AUDIT_V45.md"):
        path.write_text(audit_markdown, encoding="utf-8")
    for path in (OUT / "STEP2_DIRECT_MULTI_ACTUATOR_V45_REPORT.json", DOCS / "STEP2_DIRECT_MULTI_ACTUATOR_V45_REPORT.json"):
        path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    markdown = "# STEP2 DIRECT MULTI-ACTUATOR V4.5 REPORT\n\n```json\n" + json.dumps(report, indent=2, allow_nan=True) + "\n```\n"
    for path in (OUT / "STEP2_DIRECT_MULTI_ACTUATOR_V45_REPORT.md", DOCS / "STEP2_DIRECT_MULTI_ACTUATOR_V45_REPORT.md"):
        path.write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=True))
