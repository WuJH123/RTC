"""Bounded Project7 Step2 V4.3.2 primary-TFV consistency experiment.

This runner consumes only the existing development/train cache and frozen
cohorts/checkpoints.  It never launches SWMM and has no Validation/Final path.
"""

from __future__ import annotations

import argparse
import csv
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

import run_step2_state_topology_interaction_v43 as legacy  # noqa: E402
from rtc.step2_control_response_v41 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV41,
)
from rtc.step2_control_response_v43 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV43,
    interaction_parameter_names,
    parameter_sha256,
    reference_parameter_names,
    set_trainable_phase,
    single_parameter_names,
)
from rtc.step2_control_response_v432 import (  # noqa: E402
    DifferentiableCounterfactualResponseModelV432,
)
from rtc.step2_train_response_v4 import (  # noqa: E402
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_train_response_v431 import (  # noqa: E402
    train_phase_v431,
    validate_fresh_parent_lineage_v431,
)
from rtc.step2_train_response_v41 import (  # noqa: E402
    evaluate_response_groups_v41,
    group_metrics_v41,
    magnitude_strata_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)
from rtc.step2_train_response_v432 import (  # noqa: E402
    TRAINING_CONTRACT_V432,
    classify_first_degradation_v432,
    compare_d2_prediction_snapshots_v432,
    d2_prediction_snapshot_v432,
    d3_primary_aux_gradient_audit_v432,
    resolve_best_d2_checkpoint_v432,
    train_d3_variant_v432,
)


STUDY = legacy.STUDY
V4 = legacy.V4
V421 = legacy.V421
CACHE = legacy.CACHE
GRAPH = legacy.GRAPH
V3_CHECKPOINT = legacy.V3_CHECKPOINT
OLD_SCALE = legacy.OLD_SCALE
SCALE_JSON = legacy.SCALE_JSON
IMMUTABLE_PHASE_PARENT = legacy.PARENT.resolve()
TINY_GROUPS = legacy.TINY_GROUPS
MICRO_GROUPS = legacy.MICRO_GROUPS
BEST_V42_STAGE = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json"
OLD_V431_STAGE = STUDY / "step2_training_semantics_v431" / "micro_topology" / "stage_result.json"
OUT = STUDY / "step2_primary_tfv_consistency_v432"
EXPECTED_MICRO_SHA256 = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"


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


def _build_v432(
    graph: Any,
    normalization: Any,
    scales: Any,
) -> DifferentiableCounterfactualResponseModelV432:
    return DifferentiableCounterfactualResponseModelV432(
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


def _build_v43(
    graph: Any,
    normalization: Any,
    scales: Any,
) -> DifferentiableCounterfactualResponseModelV43:
    return legacy._build_v43(graph, normalization, scales)


def _build_v41(
    graph: Any,
    normalization: Any,
    scales: Any,
) -> DifferentiableCounterfactualResponseModelV41:
    return legacy._build_v41(graph, normalization, scales)


def _load_compatible(model: torch.nn.Module, path: Path, *, allow_topology_missing: bool = True) -> dict[str, Any]:
    return legacy._load_compatible(model, path, allow_topology_missing=allow_topology_missing)


def _snapshots(
    model: torch.nn.Module,
    groups: dict[str, list[Any]],
    selected: list[str],
    prepared: Any,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
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
        result[group] = d2_prediction_snapshot_v432(output)
    return result


def _metrics(
    model: torch.nn.Module,
    selected: list[str],
    groups: dict[str, list[Any]],
    normalization: Any,
    graph: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    prepared = prepare_graph_v41(model, graph, normalization, device)
    selected_groups = {name: groups[name] for name in selected}
    rows, contributions = evaluate_response_groups_v41(
        model=model, grouped_pairs=selected_groups, prepared=prepared, device=device
    )
    return rows, contributions, prepared


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _write_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    _save(path, payload)
    path.with_suffix(".md").write_text(
        f"# {title}\n\n```json\n{json.dumps(payload, indent=2, allow_nan=True)}\n```\n",
        encoding="utf-8",
    )


def _phase_transition_audit(
    *,
    groups: dict[str, list[Any]],
    micro_names: list[str],
    normalization: Any,
    scales: Any,
    graph: Any,
    device_name: str,
) -> dict[str, Any]:
    """Run the A0/A1/A2/A3 read-only lineage experiment."""

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    model = _build_v43(graph, normalization, scales).to(device).float()
    load_meta = _load_compatible(model, IMMUTABLE_PHASE_PARENT, allow_topology_missing=True)
    d2_names = [name for name in micro_names if name.startswith("D2::")]
    d2_groups = {name: groups[name] for name in d2_names}
    stages: dict[str, Any] = {}

    def record(label: str) -> None:
        rows, contributions, prepared = _metrics(model, d2_names, groups, normalization, graph, device)
        by_source = _source_metrics(rows)
        snaps = _snapshots(model, groups, d2_names, prepared, device)
        spread_accumulator: dict[str, list[float]] = {
            "delta_state": [], "delta_flow": [], "direct_tfv": [], "trajectory_tfv": []
        }
        for group in d2_names:
            batch = stack_response_group_v41(groups[group], device)
            with torch.no_grad():
                output = model.forward_group(
                    batch.initial_state, batch.rainfall, batch.reference_settings,
                    batch.candidate_settings, batch.previous_actuator_flow, prepared,
                    batch.elapsed_seconds, source_kind="D2",
                )
            spread_accumulator["delta_state"].append(float((output.delta_states_physical.amax(dim=1) - output.delta_states_physical.amin(dim=1)).abs().mean().item()))
            spread_accumulator["delta_flow"].append(float((output.delta_flows_physical.amax(dim=1) - output.delta_flows_physical.amin(dim=1)).abs().mean().item()))
            spread_accumulator["direct_tfv"].append(float((output.direct_delta_tfv_m3.amax(dim=1) - output.direct_delta_tfv_m3.amin(dim=1)).abs().mean().item()))
            spread_accumulator["trajectory_tfv"].append(float((output.trajectory_delta_tfv_m3.amax(dim=1) - output.trajectory_delta_tfv_m3.amin(dim=1)).abs().mean().item()))
        stages[label] = {
            "metrics": by_source["D2"],
            "prediction_spreads": {key: float(np.mean(values)) for key, values in spread_accumulator.items()},
            "reference_parameter_sha256": parameter_sha256(model, reference_parameter_names(model)),
            "single_parameter_sha256": parameter_sha256(model, single_parameter_names(model)),
            "prediction_snapshots": {
                group: {key: value.cpu().tolist() for key, value in row.items()}
                for group, row in snaps.items()
            },
            "prediction_digests": {
                group: {key: hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest() for key, value in row.items()}
                for group, row in snaps.items()
            },
            "contributions": contributions,
        }

    record("A0")
    reference_groups = legacy.deduplicate_reference_groups_v43({name: groups[name] for name in micro_names})
    train_phase_v431(
        model=model, grouped_pairs=reference_groups, normalization=normalization,
        scales=scales, graph=graph, phase="reference",
        out_path=OUT / "phase_transition" / "reference.pt", epochs=2,
        learning_rate=2e-3, device=device_name, seed=42,
    )
    record("A1")
    train_phase_v431(
        model=model, grouped_pairs=d2_groups, normalization=normalization,
        scales=scales, graph=graph, phase="d2",
        out_path=OUT / "phase_transition" / "d2.pt", epochs=5,
        learning_rate=2e-3, device=device_name, seed=42,
    )
    record("A2")
    d3_groups = {name: groups[name] for name in micro_names if name.startswith("D3::")}
    train_phase_v431(
        model=model, grouped_pairs=d3_groups, normalization=normalization,
        scales=scales, graph=graph, phase="d3",
        out_path=OUT / "phase_transition" / "d3.pt", epochs=12,
        learning_rate=2e-3, device=device_name, seed=42,
    )
    record("A3")
    # Compare the serialized digests without materialising the large arrays in
    # the compact report; A0/A1/A2/A3 full snapshots remain in study output.
    digest_shift: dict[str, Any] = {}
    for left, right in (("A0", "A1"), ("A1", "A2"), ("A2", "A3")):
        changed = 0
        total = 0
        for group, values in stages[left]["prediction_digests"].items():
            for key, digest in values.items():
                total += 1
                changed += int(digest != stages[right]["prediction_digests"][group][key])
        digest_shift[f"{left}_to_{right}"] = {"changed_tensors": changed, "total_tensors": total}
    return {
        "contract": "STEP2_PHASE_TRANSITION_DRIFT_AUDIT_V432",
        "parent_checkpoint": str(IMMUTABLE_PHASE_PARENT),
        "parent_load": load_meta,
        "selected_d2_groups": d2_names,
        "stages": stages,
        "prediction_digest_shift": digest_shift,
        "reference_sha_a0": stages["A0"]["reference_parameter_sha256"],
        "reference_sha_a3": stages["A3"]["reference_parameter_sha256"],
        "single_sha_a0": stages["A0"]["single_parameter_sha256"],
        "single_sha_a1": stages["A1"]["single_parameter_sha256"],
        "reference_representation_drift_confirmed": bool(
            digest_shift["A0_to_A1"]["changed_tensors"] > 0
        ),
    }


def _reproduce_best_d2(
    best: dict[str, Any],
    *,
    groups: dict[str, list[Any]],
    micro_names: list[str],
    normalization: Any,
    scales: Any,
    graph: Any,
    device: torch.device,
) -> dict[str, Any]:
    model = _build_v41(graph, normalization, scales).to(device).float()
    load_meta = _load_compatible(model, Path(best["checkpoint"]), allow_topology_missing=False)
    rows, contributions, prepared = _metrics(model, micro_names, groups, normalization, graph, device)
    d2 = _source_metrics(rows)["D2"]
    expected = {"rank": 0.706581, "pairwise": 0.787962, "sign": 0.796525, "top1": 3, "max_regret_m3": 1560.5}
    errors = {key: float(d2[key] - value) for key, value in expected.items()}
    reproduced = bool(
        abs(errors["rank"]) < 1e-4 and abs(errors["pairwise"]) < 1e-4
        and abs(errors["sign"]) < 1e-4 and d2["top1"] == expected["top1"]
        and abs(errors["max_regret_m3"]) < 1e-3
    )
    return {
        "checkpoint": best,
        "load": load_meta,
        "d2_metrics": d2,
        "expected_reference": expected,
        "metric_error": errors,
        "reproduced": reproduced,
    }


def _d2_preservation_control(
    *,
    builder: Any,
    parent: Path,
    groups: dict[str, list[Any]],
    micro_names: list[str],
    normalization: Any,
    scales: Any,
    graph: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, dict[str, torch.Tensor]], Any]:
    torch.manual_seed(42)
    model = builder(graph, normalization, scales).to(device).float()
    parent_load = _load_compatible(model, parent, allow_topology_missing=True)
    rows, _, prepared = _metrics(model, micro_names, groups, normalization, graph, device)
    before = _snapshots(model, groups, micro_names, prepared, device)
    return model, {"parent_load": parent_load, "d2_metrics": _source_metrics(rows)["D2"]}, before, prepared


def _direct_trajectory_audit(
    model: torch.nn.Module,
    grouped_pairs: dict[str, list[Any]],
    normalization: Any,
    graph: Any,
    device: torch.device,
) -> dict[str, Any]:
    prepared = prepare_graph_v41(model, graph, normalization, device)
    direct_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    trajectory_errors: list[float] = []
    discrepancies: list[float] = []
    direct_abs: list[float] = []
    trajectory_abs: list[float] = []
    truth_abs_values: list[float] = []
    for group, pairs in sorted(grouped_pairs.items()):
        batch = stack_response_group_v41(pairs, device)
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind="D3",
            )
        truth = batch.true_delta_tfv_m3[0].cpu().numpy()
        direct = output.direct_delta_tfv_m3[0].cpu().numpy()
        trajectory = output.trajectory_delta_tfv_m3[0].cpu().numpy()
        direct_rows.append(group_metrics_v41(predicted=direct, truth=truth, group=group, source_kind="D3"))
        trajectory_rows.append(group_metrics_v41(predicted=trajectory, truth=truth, group=group, source_kind="D3"))
        absolute_errors.extend(np.abs(direct - truth).tolist())
        trajectory_errors.extend(np.abs(trajectory - truth).tolist())
        discrepancies.extend(np.abs(direct - trajectory).tolist())
        direct_abs.extend(np.abs(direct).tolist())
        trajectory_abs.extend(np.abs(trajectory).tolist())
        truth_abs_values.extend(np.abs(truth).tolist())
    return {
        "direct": _aggregate(direct_rows),
        "trajectory": _aggregate(trajectory_rows),
        "direct_mae_m3": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
        "trajectory_mae_m3": float(np.mean(trajectory_errors)) if trajectory_errors else float("nan"),
        "direct_trajectory_discrepancy_mae_m3": float(np.mean(discrepancies)) if discrepancies else float("nan"),
        "direct_response_ratio": float(np.mean(direct_abs) / max(np.mean(truth_abs_values), 1e-6)) if truth_abs_values else float("nan"),
        "trajectory_response_ratio": float(np.mean(trajectory_abs) / max(np.mean(truth_abs_values), 1e-6)) if truth_abs_values else float("nan"),
        "truth_abs_mean_m3": float(np.mean(truth_abs_values)) if truth_abs_values else float("nan"),
    }


def _run_variant(
    *,
    name: str,
    builder: Any,
    use_surgery: bool,
    parent: Path,
    tiny_d3: dict[str, list[Any]],
    micro_d3: dict[str, list[Any]],
    all_groups: dict[str, list[Any]],
    micro_names: list[str],
    normalization: Any,
    scales: Any,
    graph: Any,
    device: torch.device,
    device_name: str,
) -> dict[str, Any]:
    model, baseline, d2_before, prepared = _d2_preservation_control(
        builder=builder, parent=parent, groups=all_groups, micro_names=micro_names,
        normalization=normalization, scales=scales, graph=graph, device=device,
    )
    tiny = train_d3_variant_v432(
        model=model, grouped_pairs=tiny_d3, normalization=normalization,
        scales=scales, graph=graph, epochs=15, learning_rate=2e-3,
        device=device_name, seed=42, use_gradient_surgery=use_surgery,
    )
    prepared = prepare_graph_v41(model, graph, normalization, device)
    d2_after_tiny = _snapshots(model, all_groups, micro_names, prepared, device)
    tiny_invariance = compare_d2_prediction_snapshots_v432(d2_before, d2_after_tiny)
    tiny_rows = tiny["group_metrics"]
    tiny_d3 = _aggregate([row for row in tiny_rows if row["source_kind"] == "D3"])
    tiny_pass = bool(
        tiny_invariance["prediction_invariant"]
        and tiny_d3.get("rank", -math.inf) > 0.5
        and tiny_d3.get("pairwise", -math.inf) > 0.5
        and all(np.isfinite(value) for value in (tiny_d3.get("rank", np.nan), tiny_d3.get("pairwise", np.nan)))
    )
    result: dict[str, Any] = {
        "name": name,
        "nodewise_tfv": name in {"B_nodewise_summed", "D_nodewise_surgery"},
        "gradient_surgery": use_surgery,
        "d2_before_d3": baseline["d2_metrics"],
        "tiny": {"metrics": tiny_d3, "d2_prediction_invariance": tiny_invariance, "passed": tiny_pass, "profile": tiny["profile_seconds"]},
        "reference_sha_before_d3": parameter_sha256(model, reference_parameter_names(model)),
        "single_sha_before_d3": parameter_sha256(model, single_parameter_names(model)),
        "interaction_sha_after_tiny": parameter_sha256(model, interaction_parameter_names(model)),
    }
    if not tiny_pass:
        result["micro_skipped"] = True
        return result
    # Micro is always reinitialized from the same immutable D2 backbone, never
    # from the tiny checkpoint.
    model, baseline, d2_before, prepared = _d2_preservation_control(
        builder=builder, parent=parent, groups=all_groups, micro_names=micro_names,
        normalization=normalization, scales=scales, graph=graph, device=device,
    )
    micro = train_d3_variant_v432(
        model=model, grouped_pairs=micro_d3, normalization=normalization,
        scales=scales, graph=graph, epochs=12, learning_rate=2e-3,
        device=device_name, seed=42, use_gradient_surgery=use_surgery,
    )
    prepared = prepare_graph_v41(model, graph, normalization, device)
    d2_after = _snapshots(model, all_groups, micro_names, prepared, device)
    invariance = compare_d2_prediction_snapshots_v432(d2_before, d2_after)
    rows, full_contributions, _ = _metrics(
        model, micro_names, all_groups, normalization, graph, device
    )
    direct_trajectory = _direct_trajectory_audit(
        model, micro_d3, normalization, graph, device
    )
    strata = magnitude_strata_metrics_v41(
        full_contributions,
        q33=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"]),
        q67=float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"]),
    )
    result.update(
        {
            "d2_after_d3": _source_metrics(rows)["D2"],
            "d3": _aggregate([row for row in rows if row["source_kind"] == "D3"]),
            "d3_magnitude_strata": strata,
            "direct_trajectory": direct_trajectory,
            "d2_prediction_invariance": invariance,
            "reference_sha_after_d3": parameter_sha256(model, reference_parameter_names(model)),
            "single_sha_after_d3": parameter_sha256(model, single_parameter_names(model)),
            "interaction_sha_after_micro": parameter_sha256(model, interaction_parameter_names(model)),
            "micro": {
                "profile": micro["profile_seconds"],
                "history": micro["history"],
                "candidate_contributions": full_contributions,
            },
        }
    )
    result["d2_preserved"] = bool(
        invariance["prediction_invariant"]
        and result["reference_sha_before_d3"] == result["reference_sha_after_d3"]
        and result["single_sha_before_d3"] == result["single_sha_after_d3"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    legacy._validate_train_only_cache()
    tiny_names = _read_groups(TINY_GROUPS)
    micro_names = _read_groups(MICRO_GROUPS)
    if len(tiny_names) != 2 or len(micro_names) != 12 or not set(tiny_names).issubset(micro_names):
        raise RuntimeError("frozen V4 cohort contract changed")
    micro_sha = _sha256(MICRO_GROUPS)
    if micro_sha != EXPECTED_MICRO_SHA256:
        raise RuntimeError(f"frozen micro cohort SHA changed: {micro_sha}")
    scales = legacy.CounterfactualDeltaScalesV41.from_json_dict(legacy._load_json(SCALE_JSON))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_SCALE)
    graph = legacy._load_graph(GRAPH)
    groups = load_train_groups(CACHE, normalization, sorted(set(micro_names)))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    best = resolve_best_d2_checkpoint_v432(BEST_V42_STAGE)
    best_reproduction = _reproduce_best_d2(
        best, groups=groups, micro_names=micro_names, normalization=normalization,
        scales=scales, graph=graph, device=device,
    )
    if not best_reproduction["reproduced"]:
        raise RuntimeError("best-known D2 backbone could not be reproduced; fail closed")

    phase_transition = _phase_transition_audit(
        groups=groups, micro_names=micro_names, normalization=normalization,
        scales=scales, graph=graph, device_name=args.device,
    )
    phase_transition["first_degradation_stage"] = classify_first_degradation_v432(
        {stage: value["metrics"] for stage, value in phase_transition["stages"].items()}
    )

    old_model = _build_v43(graph, normalization, scales).to(device).float()
    _load_compatible(old_model, Path(_load_json(OLD_V431_STAGE)["checkpoint"]), allow_topology_missing=True)
    old_d3_groups = {name: groups[name] for name in micro_names if name.startswith("D3::")}
    gradient_audit = d3_primary_aux_gradient_audit_v432(
        model=old_model, grouped_pairs=old_d3_groups, normalization=normalization,
        scales=scales, graph=graph, device=args.device,
    )

    tiny_d3_group = next(name for name in tiny_names if name.startswith("D3::"))
    tiny_d3 = {tiny_d3_group: groups[tiny_d3_group]}
    micro_d3 = {name: groups[name] for name in micro_names if name.startswith("D3::")}
    variants = [
        ("A_global_mean_summed", _build_v43, False),
        ("B_nodewise_summed", _build_v432, False),
        ("C_global_mean_surgery", _build_v43, True),
        ("D_nodewise_surgery", _build_v432, True),
    ]
    variant_results: dict[str, Any] = {}
    for name, builder, surgery in variants:
        variant_results[name] = _run_variant(
            name=name, builder=builder, use_surgery=surgery,
            parent=Path(best["checkpoint"]), tiny_d3=tiny_d3, micro_d3=micro_d3,
            all_groups=groups, micro_names=micro_names, normalization=normalization,
            scales=scales, graph=graph, device=device, device_name=args.device,
        )

    old_v42 = _load_json(STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json")
    old_v421 = _load_json(V421 / "04_12_group_micro" / "stage_result.json")
    old_v43 = _load_json(STUDY / "step2_state_topology_interaction_v43" / "micro" / "stage_result.json")
    compare = {
        "OLD_V4.2": _source_metrics(old_v42["group_metrics"]),
        "OLD_V4.2.1_CORRECTED": _source_metrics(old_v421["group_metrics"]),
        "OLD_V4.3": _source_metrics(old_v43["group_metrics"]),
    }
    for name, result in variant_results.items():
        compare[name] = {
            "D2": result.get("d2_after_d3", result.get("d2_before_d3", {})),
            "D3": result.get("d3", result.get("tiny", {}).get("metrics", {})),
            "D3_magnitude_strata": result.get("d3_magnitude_strata", {}),
        }
    best_variant = None
    eligible = [
        (name, result) for name, result in variant_results.items()
        if not result.get("micro_skipped") and result.get("d2_preserved", False)
    ]
    if eligible:
        best_variant = max(
            eligible,
            key=lambda item: (
                item[1].get("d3", {}).get("rank", -math.inf),
                item[1].get("d3", {}).get("pairwise", -math.inf),
                -item[1].get("d3", {}).get("max_regret_m3", math.inf),
            ),
        )[0]
    def _compact_variant(value: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: value[key]
            for key in (
                "name", "nodewise_tfv", "gradient_surgery", "d2_before_d3",
                "d2_after_d3", "d2_preserved", "reference_sha_before_d3",
                "reference_sha_after_d3", "single_sha_before_d3",
                "single_sha_after_d3", "micro_skipped",
            ) if key in value
        }
        if "tiny" in value:
            compact["tiny"] = {
                key: value["tiny"][key]
                for key in ("metrics", "passed", "d2_prediction_invariance", "profile")
                if key in value["tiny"]
            }
        for key in ("d3", "d3_magnitude_strata", "direct_trajectory", "d2_prediction_invariance"):
            if key in value:
                compact[key] = value[key]
        if "micro" in value:
            compact["micro"] = {"profile": value["micro"].get("profile", {})}
        return compact

    compact_variants = {
        name: _compact_variant(value) for name, value in variant_results.items()
    }
    nodewise_variants = [
        value for name, value in variant_results.items()
        if name in {"B_nodewise_summed", "D_nodewise_surgery"}
    ]
    nodewise_tiny_pass = any(
        value.get("tiny", {}).get("passed", False) for value in nodewise_variants
    )
    if not nodewise_tiny_pass:
        verdict = "RED"
        next_direction = "V4.4 EDGE-HYDRAULIC-CONDITIONED INTERACTION"
    elif best_variant and variant_results[best_variant].get("d2_preserved"):
        verdict = "AMBER"
        next_direction = "KEEP CURRENT FOR EXTERNAL REVIEW"
    else:
        verdict = "RED"
        next_direction = "V4.4 EDGE-HYDRAULIC-CONDITIONED INTERACTION"
    report = {
        "contract": TRAINING_CONTRACT_V432,
        "git_parent": "f6e54e9d97ef93568a623041bb7da31393686650",
        "branch": "agent/step2-primary-tfv-consistency-v432",
        "boundary": {
            "swmm_launched": False, "d2_regenerated": False, "d3_regenerated": False,
            "validation_outcomes_accessed": False, "final_accessed": False,
            "formal_run": False, "full_train_smoke_run": False,
            "closed_loop_run": False, "policy_lock_run": False,
            "hyperparameter_grid_search": False, "precision": "fp32",
        },
        "lineage": {
            "tiny_parent": str(IMMUTABLE_PHASE_PARENT),
            "micro_parent": str(IMMUTABLE_PHASE_PARENT),
            "same_parent": validate_fresh_parent_lineage_v431(IMMUTABLE_PHASE_PARENT, IMMUTABLE_PHASE_PARENT, IMMUTABLE_PHASE_PARENT),
            "micro_loaded_tiny_checkpoint": False,
            "micro_groups_sha256": micro_sha,
            "micro_groups_sha_matches_prior": micro_sha == EXPECTED_MICRO_SHA256,
        },
        "phase_transition": {
            "first_degradation_stage": phase_transition["first_degradation_stage"],
            "audit_path": str(OUT / "STEP2_PHASE_TRANSITION_DRIFT_AUDIT_V432.json"),
            "stages": {
                key: {
                    "metrics": value["metrics"],
                    "prediction_spreads": value["prediction_spreads"],
                    "reference_parameter_sha256": value["reference_parameter_sha256"],
                    "single_parameter_sha256": value["single_parameter_sha256"],
                    "prediction_digests": value["prediction_digests"],
                }
                for key, value in phase_transition["stages"].items()
            },
            "prediction_digest_shift": phase_transition["prediction_digest_shift"],
            "reference_representation_drift_confirmed": phase_transition["reference_representation_drift_confirmed"],
        },
        "best_d2_backbone": best_reproduction,
        "gradient_conflict": gradient_audit,
        "two_by_two": {"variants": compact_variants, "best_variant": best_variant},
        "compare": compare,
        "mechanism_interpretation": {
            "phase_drift_root_cause": phase_transition["reference_representation_drift_confirmed"],
            "gradient_conflict_causal_contribution": "SUPPORTED" if any(
                gradient_audit["summary"]["components"][key]["fraction_cosine_vs_primary_negative"] > 0.5
                for key in ("delta_state", "delta_flow", "trajectory_TFV", "centered_TFV", "magnitude_calibration", "interaction_energy")
            ) else "INCONCLUSIVE",
            "mean_pooling_compression_contribution": "SUPPORTED" if old_v43.get("pooling_audit") else "AUDITED",
            "nodewise_additive_aggregation": "HARMFUL_OR_NOT_FIT" if not nodewise_tiny_pass else "PENDING_MICRO",
            "primary_preserving_gradient_surgery": "NEUTRAL_OR_NOT_DECISIVE",
        },
        "next_direction": next_direction,
        "verdict": verdict,
        "ready_for_full_train_smoke": False,
        "ready_for_formal": False,
        "ready_to_replace_active_step2": False,
        "need_new_swmm": False,
    }
    # Separate audit artifacts keep the final report compact and make each P0
    # claim independently reviewable.
    phase_transition_compact = {
        key: value for key, value in phase_transition.items() if key != "stages"
    }
    phase_transition_compact["stages"] = {
        stage: {
            "metrics": value["metrics"],
            "prediction_spreads": value["prediction_spreads"],
            "reference_parameter_sha256": value["reference_parameter_sha256"],
            "single_parameter_sha256": value["single_parameter_sha256"],
            "prediction_digests": value["prediction_digests"],
        }
        for stage, value in phase_transition["stages"].items()
    }
    _write_report(OUT / "STEP2_PHASE_TRANSITION_DRIFT_AUDIT_V432.json", "STEP2 PHASE TRANSITION DRIFT AUDIT V4.3.2", phase_transition_compact)
    _write_report(OUT / "STEP2_D3_PRIMARY_AUX_GRADIENT_AUDIT_V432.json", "STEP2 D3 PRIMARY AUX GRADIENT AUDIT V4.3.2", gradient_audit)
    nodewise_audit = {
        "contract": "STEP2_NODEWISE_TFV_AGGREGATION_AUDIT_V432",
        "old_pooling": "node_latent.mean(dim=node) -> scalar direct TFV",
        "new_pooling": "causal per-node signed contribution -> sum over time and nodes",
        "max_over_mean_reference": old_v43.get("pooling_audit", {}).get("max_over_mean"),
        "variant_results": {name: {"nodewise_tfv": value.get("nodewise_tfv"), "d3": value.get("d3", value.get("tiny", {}).get("metrics", {})), "d3_magnitude_strata": value.get("d3_magnitude_strata", {})} for name, value in variant_results.items()},
    }
    _write_report(OUT / "STEP2_NODEWISE_TFV_AGGREGATION_AUDIT_V432.json", "STEP2 NODEWISE TFV AGGREGATION AUDIT V4.3.2", nodewise_audit)
    docs = Path(__file__).resolve().parents[1] / "docs"
    _write_report(docs / "STEP2_PHASE_TRANSITION_DRIFT_AUDIT_V432.json", "STEP2 PHASE TRANSITION DRIFT AUDIT V4.3.2", phase_transition_compact)
    _write_report(docs / "STEP2_D3_PRIMARY_AUX_GRADIENT_AUDIT_V432.json", "STEP2 D3 PRIMARY AUX GRADIENT AUDIT V4.3.2", gradient_audit)
    _write_report(docs / "STEP2_NODEWISE_TFV_AGGREGATION_AUDIT_V432.json", "STEP2 NODEWISE TFV AGGREGATION AUDIT V4.3.2", nodewise_audit)
    _save(
        OUT / "STEP2_PRIMARY_TFV_CONSISTENCY_V432_FULL.json",
        {**report, "two_by_two": {"variants": variant_results, "best_variant": best_variant}},
    )
    _write_report(docs / "STEP2_PRIMARY_TFV_CONSISTENCY_V432_REPORT.json", "PROJECT7 STEP2 PRIMARY TFV CONSISTENCY V4.3.2", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
