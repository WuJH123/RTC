"""Run the bounded V4.2 D3 interaction calibration stages.

The stages are deliberately isolated from the active Step2 trainer.  They use only
the already-frozen development/train cache, keep D2 single-effect parameters out of
D3 gradients, and stop at the 12-group micro evidence stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41
from rtc.step2_train_response_v4 import build_full_train_normalization_from_checkpoint, load_train_groups
from rtc.step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    evaluate_response_groups_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
    train_response_v41,
)
from rtc.step2_training_cache import load_step2_training_cache


STUDY = Path(r"E:\RTC_sewer\Project7\study_v069")
V4 = STUDY / "step2_control_response_v4"
V41 = STUDY / "step2_response_calibration_v41"
OUT = STUDY / "step2_d3_magnitude_calibration_v42"
CACHE = STUDY / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
OLD_SCALE = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
SCALE_JSON = V41 / "00_scales" / "counterfactual_delta_scales_train18.json"
OLD_COMBINED = V41 / "04_tiny_combined" / "v41_tiny_combined.pt"
TINY_GROUPS = V4 / "02_tiny_overfit" / "02_tiny_overfit.groups.csv"
MICRO_GROUPS = V4 / "03_12_group_micro" / "03_12_group_micro.groups.csv"

STAGES = {
    "tiny-d3": ("02_tiny_d3", "v42_tiny_d3.pt", 100),
    "tiny-combined": ("03_tiny_combined", "v42_tiny_combined.pt", 60),
    "micro": ("04_12_group_micro", "v42_12_group_micro.pt", 30),
}


def _read_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def _sha256_state(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(prefixes):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _build_model(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41, *, magnitude: bool) -> DifferentiableCounterfactualResponseModelV41:
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
        interaction_magnitude_features_enabled=magnitude,
    )


def _load_compatible(model: DifferentiableCounterfactualResponseModelV41, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in incoming.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed_missing = tuple(name for name in missing if name.startswith("interaction_"))
    if unexpected or len(allowed_missing) != len(missing):
        raise RuntimeError(f"V4.2 parent mismatch outside interaction branch: missing={missing} unexpected={unexpected}")


def _validate_cache() -> None:
    cache = load_step2_training_cache(CACHE / "CACHE_MANIFEST.json")
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
        raise RuntimeError(f"V4.2 cache is not Train-only: split={split} fold={fold}")


def _d2_equivalence(
    graph: Any,
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    pairs: list[Any],
    parent: Path,
    device: torch.device,
) -> dict[str, Any]:
    baseline = _build_model(graph, normalization, scales, magnitude=False).to(device).float()
    candidate = _build_model(graph, normalization, scales, magnitude=True).to(device).float()
    _load_compatible(baseline, parent)
    _load_compatible(candidate, parent)
    baseline.eval(); candidate.eval()
    baseline_prepared = prepare_graph_v41(baseline, graph, normalization, device)
    candidate_prepared = prepare_graph_v41(candidate, graph, normalization, device)
    batch = stack_response_group_v41(pairs, device)
    with torch.no_grad():
        base = baseline.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, baseline_prepared, batch.elapsed_seconds, source_kind="D2")
        new = candidate.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings, batch.previous_actuator_flow, candidate_prepared, batch.elapsed_seconds, source_kind="D2")
    diffs = {
        "single_delta_states_max_abs": float((base.single_delta_states_physical - new.single_delta_states_physical).abs().max()),
        "single_delta_flows_max_abs": float((base.single_delta_flows_physical - new.single_delta_flows_physical).abs().max()),
        "direct_single_tfv_max_abs": float((base.direct_single_delta_tfv_m3 - new.direct_single_delta_tfv_m3).abs().max()),
    }
    diffs["exact_unchanged"] = all(value == 0.0 for value in diffs.values())
    return diffs


def _gradient_diagnostic(model: torch.nn.Module, pairs: list[Any], prepared: Any, device: torch.device) -> dict[str, float]:
    batch = stack_response_group_v41(pairs, device)
    candidate = batch.candidate_settings.detach().clone().requires_grad_(True)
    output = model.forward_group(batch.initial_state, batch.rainfall, batch.reference_settings, candidate, batch.previous_actuator_flow, prepared, batch.elapsed_seconds, source_kind=batch.source_kind)
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0]
    changed = (candidate.detach() - batch.reference_settings[:, None]).abs() > 1e-8
    return {
        "finite_fraction": float(torch.isfinite(gradient).float().mean()),
        "nonzero_fraction": float(gradient.ne(0.0).float().mean()),
        "changed_action_nonzero_fraction": float(gradient[changed].ne(0.0).float().mean()),
        "l2_norm": float(torch.linalg.vector_norm(gradient)),
    }


def run_stage(
    stage: str, *, device_name: str = "cuda", output_root: Path = OUT
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    _validate_cache()
    tiny_groups = _read_groups(TINY_GROUPS)
    micro_groups = _read_groups(MICRO_GROUPS)
    d2_tiny = [group for group in tiny_groups if group.startswith("D2::")]
    d3_tiny = [group for group in tiny_groups if group.startswith("D3::")]
    if len(d2_tiny) != 1 or len(d3_tiny) != 1 or len(micro_groups) != 12:
        raise RuntimeError("V4.2 frozen cohort contract changed")
    if not set(tiny_groups).issubset(micro_groups):
        raise RuntimeError("V4.2 tiny cohort is not contained in the frozen micro cohort")
    if stage == "tiny-d3":
        selected = d3_tiny
        parent = OLD_COMBINED
    elif stage == "tiny-combined":
        selected = d2_tiny + d3_tiny
        parent = output_root / STAGES["tiny-d3"][0] / STAGES["tiny-d3"][1]
    else:
        selected = micro_groups
        parent = output_root / STAGES["tiny-combined"][0] / STAGES["tiny-combined"][1]
    if not parent.is_file():
        raise RuntimeError(f"V4.2 preceding checkpoint missing: {parent}")
    scales = CounterfactualDeltaScalesV41.from_json_dict(json.loads(SCALE_JSON.read_text(encoding="utf-8")))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = _load_graph(GRAPH)
    groups = load_train_groups(CACHE, normalization, sorted(set(selected + d2_tiny)))
    training_groups = {group: groups[group] for group in selected}
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_model(graph, normalization, scales, magnitude=True).to(device).float()
    _load_compatible(model, parent)
    prepared = prepare_graph_v41(model, graph, normalization, device)
    d2_equivalence = _d2_equivalence(graph, normalization, scales, groups[d2_tiny[0]], parent, device)
    single_before = _sha256_state(model, ("single_", "direct_single_tfv_head"))
    output_path = output_root / STAGES[stage][0] / STAGES[stage][1]
    training = train_response_v41(
        model=model,
        grouped_pairs=training_groups,
        normalization=normalization,
        scales=scales,
        graph=graph,
        out_path=output_path,
        epochs=STAGES[stage][2],
        learning_rate=2e-3,
        seed=42,
        device=device_name,
        early_stop_patience=25,
        magnitude_calibration=True,
        selection_policy=("rank_first" if stage in {"tiny-d3", "tiny-combined"} else "d3_magnitude"),
    )
    single_after = _sha256_state(model, ("single_", "direct_single_tfv_head"))
    prepared = prepare_graph_v41(model, graph, normalization, device)
    metric_rows, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=training_groups, prepared=prepared, device=device
    )
    d2_metrics = [row for row in metric_rows if row["source_kind"] == "D2"]
    d3_metrics = [row for row in metric_rows if row["source_kind"] == "D3"]
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    d3_strata = magnitude_strata_metrics_v41(contributions, q33=float(q["q33"]), q67=float(q["q67"]))
    gradient_pairs = groups[d3_tiny[0]] if d3_tiny[0] in groups else groups[selected[0]]
    result = {
        "contract": "STEP2_D3_MAGNITUDE_CALIBRATION_V42_STAGE_RESULT",
        "stage": stage,
        "boundary": {
            "scientific_split": ["development"],
            "development_fold": ["train"],
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_step2_run": False,
            "closed_loop_run": False,
            "full_train_smoke_run": False,
            "precision": "fp32",
            "hyperparameter_grid_search": False,
        },
        "selected_groups": selected,
        "parent_checkpoint": str(parent),
        "training": training,
        "group_metrics": metric_rows,
        "d3_magnitude_strata": d3_strata,
        "d2_tiny_baseline_equivalence_before_update": d2_equivalence,
        "single_branch_parameter_sha_before": single_before,
        "single_branch_parameter_sha_after": single_after,
        "d2_single_branch_parameters_unchanged": single_before == single_after,
        "gradient_diagnostic": _gradient_diagnostic(model, gradient_pairs, prepared, device),
        "scale_manifest_sha256": scales.source_manifest_sha256,
        "interaction_magnitude_features_enabled": True,
        "multi_timescale_memory_enabled": False,
        "fixed_memory_rho": 0.65,
        "next_bounded_action": "Proceed only to the next frozen V4.2 stage if D2 remains non-degraded and D3 large/rank evidence improves.",
    }
    result_path = output_root / STAGES[stage][0] / "stage_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage": stage,
        "d2_groups": len(d2_metrics),
        "d3_groups": len(d3_metrics),
        "d2_single_branch_parameters_unchanged": result["d2_single_branch_parameters_unchanged"],
        "d2_equivalence_before_update": d2_equivalence,
        "d3_strata": d3_strata,
        "gradient": result["gradient_diagnostic"],
        "output": str(result_path),
    }, indent=2, allow_nan=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    run_stage(args.stage, device_name=args.device, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
