"""Compile a self-describing, development-only V11 runtime bundle.

This command does not train, run SWMM, access Validation/Final, or promote a
policy. It reconstructs the exact TrainFit normalization/scales from the frozen
D2+D3 cache and binds them to an already trained D3 V11 checkpoint after
fail-closed held-out development checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_objective_v110 import derive_effect_scales_v110
from rtc.step2_mpc_v110 import RuntimeNormalizationV110
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_v110_contract import V110_CONTRACT
from rtc.step2_runtime_v110 import V110_RUNTIME_BUNDLE_CONTRACT


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _positive_primary(report: dict[str, Any]) -> dict[str, float]:
    raw = report.get("holdout_primary_skill")
    if not isinstance(raw, dict):
        raise ValueError("V11 D3 report lacks holdout_primary_skill")
    required = ("depth", "flood", "volume", "managed_flow")
    values = {name: float(raw.get(name, float("nan"))) for name in required}
    if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError(f"V11 D3 held-out primary skills are not all positive: {values}")
    return values


def _normalization_payload(norm) -> dict[str, list[float]]:
    runtime = RuntimeNormalizationV110(
        state_mean=np.asarray(norm.state_mean, dtype=np.float32),
        state_std=np.asarray(norm.state_std, dtype=np.float32),
        rainfall_mean=np.asarray(norm.rainfall_mean, dtype=np.float32),
        rainfall_std=np.asarray(norm.rainfall_std, dtype=np.float32),
        flow_mean=np.asarray(norm.flow_mean, dtype=np.float32),
        flow_std=np.asarray(norm.flow_std, dtype=np.float32),
    )
    runtime.validate()
    return runtime.as_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Project7 V11 development runtime bundle")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--v110-d3-checkpoint", required=True)
    parser.add_argument("--v110-d3-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        raise ValueError("V11 runtime bundle requires the frozen 932-node/109-actuator graph")
    cache = V60TrainCache(args.cache_manifest)
    lineage_guard = validate_v60_cache_lineage(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V11 runtime bundle refuses legacy dense D3")
    d2 = cache.names("D2")
    d3 = cache.targeted_d3_names()
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(d2 + d3), holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V11 runtime bundle split lacks a D2/D3 side")

    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V11 runtime bundle detected split leakage")

    d3_report = _load_json(args.v110_d3_report)
    if d3_report.get("contract") != V110_CONTRACT or d3_report.get("stage") != "d3":
        raise ValueError("runtime bundle requires a V11 D3 development report")
    d3_primary = _positive_primary(d3_report)
    report_lineage = d3_report.get("lineage")
    if not isinstance(report_lineage, dict):
        raise ValueError("V11 D3 report lacks lineage")
    expected = {
        "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "v70_value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
        "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
    }
    for key, value in expected.items():
        if str(report_lineage.get(key, "")) != value:
            raise ValueError(f"V11 D3 report lineage mismatch: {key}")

    checkpoint = torch.load(args.v110_d3_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("V11 D3 checkpoint must be a dictionary")
    if checkpoint.get("contract") != V110_CONTRACT or checkpoint.get("stage") != "d3":
        raise ValueError("runtime bundle requires a V11 D3 checkpoint")
    if "state_dict" not in checkpoint:
        raise ValueError("V11 D3 checkpoint lacks state_dict")
    checkpoint_lineage = checkpoint.get("lineage")
    if not isinstance(checkpoint_lineage, dict):
        raise ValueError("V11 D3 checkpoint lacks lineage")
    for key, value in expected.items():
        if str(checkpoint_lineage.get(key, "")) != value:
            raise ValueError(f"V11 D3 checkpoint lineage mismatch: {key}")

    normalization = derive_input_normalization_v60(cache, fit)
    prepared = prepare_static_v60(graph, "cpu")
    scales = derive_effect_scales_v110(cache, fit_d2, prepared)
    state_dict = checkpoint["state_dict"]
    state_scale = np.asarray(state_dict["state_magnitude_scale"], dtype=np.float32)
    flow_scale = np.asarray(state_dict["flow_magnitude_scale"], dtype=np.float32)
    if not np.allclose(state_scale, scales.state_magnitude_scale, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError("V11 checkpoint state scale differs from reconstructed TrainFit scale")
    if not np.allclose(flow_scale, scales.flow_magnitude_scale, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError("V11 checkpoint flow scale differs from reconstructed TrainFit scale")

    source_manifest = _load_json(lineage_guard["source_manifest"])
    engine = str(source_manifest.get("swmm_engine_version", "")).strip()
    if not engine:
        raise ValueError("V11 source shard manifest lacks SWMM engine lineage")
    if int(source_manifest.get("model_step_seconds", -1)) != 300:
        raise ValueError("V11 runtime bundle source step is not 300 s")
    if int(source_manifest.get("horizon_steps", -1)) != 72:
        raise ValueError("V11 runtime bundle source horizon is not 72 steps")

    output = dict(checkpoint)
    output.update(
        {
            "runtime_bundle_contract": V110_RUNTIME_BUNDLE_CONTRACT,
            "runtime_compatible": True,
            "production_compatible": False,
            "input_normalization": _normalization_payload(normalization),
            "effect_thresholds": {
                "state_active_threshold": scales.state_active_threshold.astype(float).tolist(),
                "flow_active_threshold": scales.flow_active_threshold.astype(float).tolist(),
                "source_group_count": int(scales.source_group_count),
            },
            "swmm_engine_version": engine,
            "model_step_seconds": 300,
            "value_horizon_steps": 72,
            "hydraulic_horizon_minutes": 120,
            "control_update_seconds": 600,
            "development_gate": {
                "d3_primary_skill_positive": True,
                "holdout_primary_skill": d3_primary,
                "event_overlap": 0,
                "rainfall_overlap": 0,
            },
            "runtime_lineage": {
                **expected,
                "source_shard_manifest_sha256": str(lineage_guard["source_manifest_sha256"]),
                "v60_control_basis_sha256": str(lineage_guard["v60_control_basis_sha256"]),
                "v60_design_contract_sha256": str(lineage_guard["v60_design_contract_sha256"]),
                "v110_d3_report_sha256": _sha256(args.v110_d3_report),
                "v110_d3_checkpoint_sha256": _sha256(args.v110_d3_checkpoint),
            },
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    audit = {
        "contract": V110_RUNTIME_BUNDLE_CONTRACT,
        "bundle": str(out.resolve()),
        "bundle_sha256": _sha256(out),
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "runtime_compatible": True,
        "production_compatible": False,
        "d3_primary_skill": d3_primary,
        "fit_group_count": len(fit),
        "holdout_group_count": len(holdout),
        "fit_d2_group_count": len(fit_d2),
        "fit_d3_group_count": len(fit_d3),
        "event_overlap": 0,
        "rainfall_overlap": 0,
        "swmm_engine_version": engine,
        "lineage": output["runtime_lineage"],
    }
    audit_path = out.with_suffix(out.suffix + ".json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
