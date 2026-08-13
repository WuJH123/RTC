"""Run the bounded V9 conduit-physical-edge D2 mechanism control.

This diagnostic is deliberately separate from the A/B/C state-sufficiency
runner.  It uses only the canonical TrainFit D2 groups, the frozen V7 reference
Hydraulic model, and predicted (never authoritative/oracle) reference
trajectories.  It neither invokes SWMM nor admits D3, Validation, Final, or
Formal assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import HydraulicResponseSurrogateV70
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_hydraulic_eval_v90 import evaluate_hydraulic_effect_v90
from rtc.step2_hydraulic_objective_v90 import derive_onset_sqrt_positive_weight_v80
from rtc.step2_optimization_v90 import candidate_batch_chunks_v90, train_d2_mechanism_v90
from rtc.step2_physical_edge_v90 import (
    ConduitPhysicalEdgeAssetsV90,
    build_conduit_physical_edge_assets_v90,
)
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v70_contract import V70_CONTRACT
from rtc.step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_B, V90_CONTRACT


_CANONICAL_SEED = 42
_CANONICAL_HOLDOUT_FRACTION = 0.20
# Execution-only memory guard for preflight probes.  Training preserves the
# original full-group loss exactly; the physical message operator itself uses
# activation recomputation to fit the frozen full 24-candidate objective.
_EXECUTION_CANDIDATE_CHUNK_SIZE = 4


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


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


def _load_checkpoint(path: str | Path, expected_kind: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a dictionary")
    if str(payload.get("contract")) != V70_CONTRACT:
        raise ValueError(f"checkpoint {path} is not a V7 canonical checkpoint")
    if str(payload.get("kind")) != expected_kind:
        raise ValueError(f"checkpoint kind {payload.get('kind')} != {expected_kind}")
    if "state_dict" not in payload:
        raise ValueError("V7 checkpoint has no state_dict")
    return payload


def _require_equal(label: str, values: list[str]) -> str:
    cleaned = [str(value) for value in values]
    if not all(cleaned) or len(set(cleaned)) != 1:
        raise ValueError(f"V9 physical-edge lineage mismatch for {label}: {cleaned}")
    return cleaned[0]


def _validate_v7_lineage(
    *,
    value_checkpoint: dict[str, Any],
    hydraulic_checkpoint: dict[str, Any],
    graph_sha256: str,
    cache_manifest_sha256: str,
    cache_lineage: dict[str, Any],
) -> dict[str, str]:
    """Fail closed unless both frozen V7 checkpoints admit this exact cache."""
    value = value_checkpoint.get("lineage", {})
    hydraulic = hydraulic_checkpoint.get("lineage", {})
    if not isinstance(value, dict) or not isinstance(hydraulic, dict):
        raise ValueError("V9 physical-edge checkpoints lack V7 lineage dictionaries")
    return {
        "graph_sha256": _require_equal(
            "graph_sha256",
            [graph_sha256, value.get("graph_sha256", ""), hydraulic.get("graph_sha256", "")],
        ),
        "cache_manifest_sha256": _require_equal(
            "cache_manifest_sha256",
            [
                cache_manifest_sha256,
                value.get("cache_manifest_sha256", ""),
                hydraulic.get("cache_manifest_sha256", ""),
            ],
        ),
        "basis_sha256": _require_equal(
            "basis_sha256",
            [
                cache_lineage.get("v60_control_basis_sha256", ""),
                value.get("basis_sha256_from_cache_lineage", ""),
                hydraulic.get("basis_sha256_from_cache_lineage", ""),
            ],
        ),
        "design_sha256": _require_equal(
            "design_sha256",
            [
                cache_lineage.get("v60_design_contract_sha256", ""),
                value.get("design_sha256_from_cache_lineage", ""),
                hydraulic.get("design_sha256_from_cache_lineage", ""),
            ],
        ),
        "split_manifest_sha256": _require_equal(
            "split_manifest_sha256",
            [
                value_checkpoint.get("split_manifest_sha256", ""),
                hydraulic_checkpoint.get("split_manifest_sha256", ""),
            ],
        ),
    }


def _validate_canonical_schedule(*, seed: int, holdout_fraction: float) -> None:
    if int(seed) != _CANONICAL_SEED:
        raise ValueError("V9 physical-edge seed is frozen at 42")
    if abs(float(holdout_fraction) - _CANONICAL_HOLDOUT_FRACTION) > 1e-12:
        raise ValueError("V9 physical-edge split is frozen at 0.20")


def _assert_full_horizon_causality(
    *,
    candidate_settings: torch.Tensor,
    baseline_output: Any,
    forward: Callable[[torch.Tensor], Any],
) -> None:
    """Prove that every retained state and managed-flow output is prefix causal."""
    retained = baseline_output.horizon_indices.detach().cpu().tolist()
    if not retained:
        raise RuntimeError("V9 physical-edge retained horizon is empty")
    baseline_state = baseline_output.raw_delta_states_physical.detach()
    baseline_flow = baseline_output.raw_delta_flows_physical.detach()
    for output_index, raw_index in enumerate(retained):
        cutoff = int(raw_index) + 1
        if cutoff >= candidate_settings.shape[2]:
            continue
        later = candidate_settings.detach().clone()
        later[:, :, cutoff:] += 0.123
        with torch.no_grad():
            changed = forward(later)
        if not torch.equal(
            baseline_state[:, :, output_index],
            changed.raw_delta_states_physical[:, :, output_index],
        ):
            raise RuntimeError(
                f"future candidate action affected retained V9 physical-edge state at output {output_index}"
            )
        if not torch.equal(
            baseline_flow[:, :, output_index],
            changed.raw_delta_flows_physical[:, :, output_index],
        ):
            raise RuntimeError(
                f"future candidate action affected retained V9 physical-edge flow at output {output_index}"
            )


def _build_reference(
    graph: GraphSchema,
    cache: V60TrainCache,
    fit_names: list[str],
    scales: Any,
) -> HydraulicResponseSurrogateV70:
    entry = cache.entry(fit_names[0])
    prepared = prepare_static_v60(graph, "cpu")
    return HydraulicResponseSurrogateV70(
        state_scale=scales.base.state_scale,
        flow_scale=scales.base.flow_scale,
        horizon_contract=MultiResolutionHorizonV60(),
        state_dim=int(entry.arrays["initial_state"].shape[-1]),
        rainfall_dim=int(entry.arrays["rainfall"].shape[-1]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        hidden_dim=64,
        latent_dim=32,
        temporal_dim=12,
    )


def _construct_physical_edge_model(
    model_type: Callable[..., Any],
    *,
    base_kwargs: dict[str, Any],
    physical_edge_assets: ConduitPhysicalEdgeAssetsV90,
) -> Any:
    """Require explicit physical-asset admission; never silently use topology only."""
    try:
        return model_type(**base_kwargs, physical_edge_assets=physical_edge_assets)
    except TypeError as exc:
        text = str(exc)
        if "physical_edge_assets" in text or "unexpected keyword" in text:
            raise TypeError(
                "V9 physical-edge runner requires a model that explicitly accepts "
                "physical_edge_assets"
            ) from exc
        raise


def _physical_edge_lineage(assets: ConduitPhysicalEdgeAssetsV90) -> dict[str, Any]:
    conduit_only = (
        int(assets.regulator_propagation_edge_count) == 0
        and not bool(assets.uses_future_truth)
        and not bool(assets.uses_online_link_flow)
    )
    return {
        "contract": assets.contract,
        "frozen_inp_path": assets.inp_path,
        "frozen_inp_sha256": assets.inp_sha256,
        "physical_link_count": int(assets.physical_link_count),
        "conduit_physical_link_count": int(assets.conduit_physical_link_count),
        "directed_conduit_edge_count": int(assets.directed_edge_count),
        "static_normalization_sha256": assets.static_normalization_sha256,
        "regulator_propagation_edge_count": int(assets.regulator_propagation_edge_count),
        "uses_future_truth": bool(assets.uses_future_truth),
        "uses_online_link_flow": bool(assets.uses_online_link_flow),
        "conduit_only": conduit_only,
    }


def _physical_dynamic_scales(
    normalization: Any,
    assets: ConduitPhysicalEdgeAssetsV90,
) -> dict[str, float | str]:
    """Freeze dynamic physical-edge scaling without inspecting effect targets.

    Head/depth units come from the canonical TrainFit input-state normalizer.
    The gradient scale combines the TrainFit head scale with the median frozen
    conduit length.  This is deterministic, causal, and cannot leak a D2/D3
    counterfactual effect label into a message feature.
    """
    state_std = np.asarray(normalization.state_std, dtype=np.float64).reshape(-1)
    lengths = np.asarray(assets.edge_length_m, dtype=np.float64).reshape(-1)
    if state_std.size < 2 or not np.isfinite(state_std[:2]).all():
        raise ValueError("V9 physical-edge requires finite TrainFit depth/head input scales")
    if lengths.size == 0 or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("V9 physical-edge requires positive finite frozen conduit lengths")
    depth_scale = max(float(abs(state_std[0])), 1e-6)
    head_scale = max(float(abs(state_std[1])), 1e-6)
    median_length = max(float(np.median(lengths)), 1e-6)
    return {
        "head_scale_m": head_scale,
        "depth_scale_m": depth_scale,
        "gradient_scale": head_scale / median_length,
        "source": "TrainFit input-state std plus frozen conduit median length; no effect targets",
    }


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    members = (
        "scripts/run_step2_v90_physical_edge_d2.py",
        "src/rtc/step2_control_response_v90.py",
        "src/rtc/step2_physical_edge_v90.py",
        "src/rtc/step2_hydraulic_objective_v90.py",
        "src/rtc/step2_hydraulic_eval_v90.py",
        "src/rtc/step2_optimization_v90.py",
    )
    digest = hashlib.sha256()
    for relative in members:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _forward(model: Any, batch: Any, prepared: Any):
    return model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        batch.candidate_settings,
        batch.previous_actuator_flow,
        prepared,
    )


def _preflight(
    model: Any,
    cache: V60TrainCache,
    name: str,
    normalization: Any,
    graph: GraphSchema,
    device: str,
    assets: ConduitPhysicalEdgeAssetsV90,
) -> dict[str, Any]:
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    if assets.node_count != len(graph.node_ids):
        raise ValueError("physical-edge asset node count differs from graph")
    if assets.directed_edge_count != 2 * assets.conduit_physical_link_count:
        raise ValueError("physical-edge asset did not preserve two directions per conduit")
    if assets.regulator_propagation_edge_count != 0:
        raise ValueError("regulated links entered conduit-only physical propagation")
    if assets.uses_future_truth or assets.uses_online_link_flow:
        raise ValueError("physical-edge assets violate causal V9 contract")

    model.to(target).eval()
    prepared = prepare_static_v80(graph, target)
    batch = cache.batch(name, normalization, target)
    with torch.no_grad():
        zero = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.reference_settings[:, None],
            batch.previous_actuator_flow,
            prepared,
        )
    if not torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical)):
        raise RuntimeError("V9 physical-edge exact-zero signed state effect failed")
    if not torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical)):
        raise RuntimeError("V9 physical-edge exact-zero signed flow effect failed")
    if any(parameter.requires_grad for parameter in model.reference_model.parameters()):
        raise RuntimeError("V9 physical-edge runner found trainable frozen V7 Hydraulic parameters")

    gradient_total = 0
    gradient_finite = 0
    gradient_nonzero = 0
    for chunk in candidate_batch_chunks_v90(
        batch, candidate_chunk_size=_EXECUTION_CANDIDATE_CHUNK_SIZE
    ):
        candidate = chunk.candidate_settings.detach().clone().requires_grad_(True)
        output = model(
            chunk.initial_state,
            chunk.rainfall,
            chunk.reference_settings,
            candidate,
            chunk.previous_actuator_flow,
            prepared,
        )
        score = (
            output.raw_delta_states_physical.square().mean()
            + output.raw_delta_flows_physical.square().mean()
        )
        (gradient,) = torch.autograd.grad(score, candidate, allow_unused=False)
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("V9 physical-edge action gradient is nonfinite")
        if int(torch.count_nonzero(gradient)) == 0:
            raise RuntimeError("V9 physical-edge action gradient is identically zero")
        _assert_full_horizon_causality(
            candidate_settings=candidate,
            baseline_output=output,
            forward=lambda settings: model(
                chunk.initial_state,
                chunk.rainfall,
                chunk.reference_settings,
                settings,
                chunk.previous_actuator_flow,
                prepared,
            ),
        )
        gradient_total += int(gradient.numel())
        gradient_finite += int(torch.isfinite(gradient).sum().item())
        gradient_nonzero += int(torch.count_nonzero(gradient).item())

    return {
        "signed_state_exact_zero": True,
        "signed_flow_exact_zero": True,
        "reference_frozen": True,
        "physical_asset_contract": _physical_edge_lineage(assets),
        "future_action_causality": True,
        "action_gradient_finite_fraction": float(gradient_finite / max(gradient_total, 1)),
        "action_gradient_nonzero_fraction": float(gradient_nonzero / max(gradient_total, 1)),
    }


def _build_model(
    *,
    graph: GraphSchema,
    cache: V60TrainCache,
    fit_names: list[str],
    scales: Any,
    basis: Any,
    hydraulic_state: dict[str, torch.Tensor],
    contract: DirectHydraulicEffectLossContractV90,
    seed: int,
    assets: ConduitPhysicalEdgeAssetsV90,
    dynamic_scales: dict[str, float | str],
) -> Any:
    # Import lazily so this runner can be tested before the physical model branch
    # lands; execution still fails closed when the expected class is unavailable.
    from rtc.step2_control_response_v90 import PhysicalConduitHydraulicEffectSurrogateV90

    torch.manual_seed(seed)
    np.random.seed(seed)
    reference = _build_reference(graph, cache, fit_names, scales)
    reference.load_state_dict(hydraulic_state, strict=True)
    prepared = prepare_static_v60(graph, "cpu")
    return _construct_physical_edge_model(
        PhysicalConduitHydraulicEffectSurrogateV90,
        base_kwargs={
            "reference_model": reference,
            "temporal_basis": basis.temporal_basis,
            "control_block_steps": basis.horizon.control_block_steps,
            "state_delta_scale": scales.state_delta_scale,
            "flow_delta_scale": scales.flow_delta_scale,
            "physics_dim": int(prepared.actuator_physics.shape[1]),
            "node_static_dim": int(graph.static_node_features.shape[1]),
            "actuator_count": len(graph.actuator_ids),
            "conditioning_level": LEVEL_B,
            "contract": contract,
            "head_scale_m": float(dynamic_scales["head_scale_m"]),
            "depth_scale_m": float(dynamic_scales["depth_scale_m"]),
            "gradient_scale": float(dynamic_scales["gradient_scale"]),
        },
        physical_edge_assets=assets,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 V9 conduit-physical-edge D2 mechanism")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--frozen-inp", required=True)
    parser.add_argument("--expected-inp-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    _validate_canonical_schedule(seed=args.seed, holdout_fraction=args.holdout_fraction)

    started = time.perf_counter()
    graph = _load_graph(args.graph)
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        raise ValueError("V9 physical-edge mechanism requires the frozen 932-node / 109-actuator graph")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V9 physical-edge mechanism refuses legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(d2 + d3), holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if len(fit_d2) != 112 or len(holdout_d2) != 32:
        raise ValueError(
            "V9 physical-edge D2 mechanism requires the canonical 112/32 TrainFit/InternalHoldout D2 split"
        )
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V9 physical-edge TrainFit/InternalHoldout leakage detected")

    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    value_checkpoint = _load_checkpoint(args.v70_value_checkpoint, "control_value")
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    v7_lineage = _validate_v7_lineage(
        value_checkpoint=value_checkpoint,
        hydraulic_checkpoint=hydraulic_checkpoint,
        graph_sha256=_sha256(args.graph),
        cache_manifest_sha256=_sha256(args.cache_manifest),
        cache_lineage=cache_lineage,
    )
    assets = build_conduit_physical_edge_assets_v90(
        inp_path=args.frozen_inp,
        expected_inp_sha256=args.expected_inp_sha256,
        node_ids=graph.node_ids,
    )
    if not _physical_edge_lineage(assets)["conduit_only"]:
        raise ValueError("V9 physical-edge mechanism failed conduit-only asset lineage")
    dynamic_scales = _physical_dynamic_scales(normalization, assets)
    contract = DirectHydraulicEffectLossContractV90()
    contract.validate()
    onset_positive_weight = derive_onset_sqrt_positive_weight_v80(
        cache, fit, normalization, contract=contract
    )
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_model(
        graph=graph,
        cache=cache,
        fit_names=fit,
        scales=scales,
        basis=basis,
        hydraulic_state=hydraulic_checkpoint["state_dict"],
        contract=contract,
        seed=args.seed,
        assets=assets,
        dynamic_scales=dynamic_scales,
    )
    preflight = _preflight(model, cache, fit_d2[0], normalization, graph, str(target), assets)
    history = train_d2_mechanism_v90(
        model,
        cache,
        fit_d2_names=fit_d2,
        normalization=normalization,
        scales=scales,
        onset_positive_weight=onset_positive_weight,
        graph=graph,
        device=str(target),
        seed=args.seed,
        contract=contract,
    )
    prepared = prepare_static_v80(graph, target)
    fit_metrics = evaluate_hydraulic_effect_v90(
        model, cache, fit_d2, normalization, prepared, scales,
        graph=graph, device=target, contract=contract,
    )
    holdout_metrics = evaluate_hydraulic_effect_v90(
        model, cache, holdout_d2, normalization, prepared, scales,
        graph=graph, device=target, contract=contract,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "PROJECT7_STEP2_V90_PHYSICAL_EDGE_D2_MECHANISM_V1",
        "development_only": True,
        "production_compatible": False,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "legacy_dense_d3_present": False,
        "oracle_reference_used": False,
        "conditioning_level": LEVEL_B,
        "lineage": {
            "git_head": _git_head(),
            "implementation_sha256": _implementation_sha256(),
            "graph_sha256": _sha256(args.graph),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
            "hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
            "value_checkpoint_contract": value_checkpoint.get("contract"),
            "cache_lineage": cache_lineage,
            "v7_lineage": v7_lineage,
            "physical_edge": _physical_edge_lineage(assets),
            "physical_dynamic_scales": dynamic_scales,
            "fit_d2_group_count": len(fit_d2),
            "holdout_d2_group_count": len(holdout_d2),
            "fit_d2_group_digest": hashlib.sha256("\n".join(sorted(fit_d2)).encode("utf-8")).hexdigest(),
            "seed": args.seed,
        },
        "diagnostic_schedule": {
            "source": "TRAINFIT_D2_ONLY",
            "epochs": int(contract.d2_pretrain_epochs),
            "optimizer": "AdamW",
            "learning_rate": float(contract.learning_rate),
            "weight_decay": float(contract.weight_decay),
            "grad_clip": float(contract.grad_clip),
            "fp32": True,
            "preflight_candidate_chunk_size": _EXECUTION_CANDIDATE_CHUNK_SIZE,
            "training_execution": "full-group frozen objective with physical-message activation recomputation",
        },
        "preflight": preflight,
        "training_history": history,
        "metrics": {"TrainFit_D2": fit_metrics, "TrainInternalHoldout_D2": holdout_metrics},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = out_dir / "STEP2_V90_PHYSICAL_EDGE_D2_MECHANISM.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "preflight": preflight}, indent=2), flush=True)


if __name__ == "__main__":
    main()
