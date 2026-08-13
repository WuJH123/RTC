"""Run the Project7 V9 D2 state-sufficiency ladder using existing Train assets only.

This runner never invokes SWMM and never reads Validation/Final/Formal assets.  It is a
development diagnostic that trains A/B/C on the exact same D2 TrainFit groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import HydraulicResponseSurrogateV70
from rtc.step2_control_response_v90 import DirectHydraulicEffectSurrogateV90, prepare_static_v80
from rtc.step2_hydraulic_eval_v90 import decide_state_sufficiency_v90, evaluate_hydraulic_effect_v90
from rtc.step2_hydraulic_objective_v90 import derive_onset_sqrt_positive_weight_v80
from rtc.step2_optimization_v90 import train_d2_mechanism_v90
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v70_contract import V70_CONTRACT
from rtc.step2_v90_contract import (
    DirectHydraulicEffectLossContractV90,
    LEVEL_A,
    LEVEL_B,
    LEVEL_C,
    V90_CONTRACT,
)


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


def _implementation_sha256() -> str:
    """Hash the exact V9 ladder implementation, including evaluation semantics.

    Git HEAD alone does not identify an accidental dirty worktree at execution time.
    The compact content digest makes the evidence self-describing without treating a
    whole repository checkout as a data artifact.
    """
    root = Path(__file__).resolve().parents[1]
    members = (
        "scripts/run_step2_v90_state_sufficiency.py",
        "src/rtc/step2_v90_contract.py",
        "src/rtc/step2_control_response_v90.py",
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


def _build_reference(graph: GraphSchema, cache: V60TrainCache, fit_names, scales):
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


def _build_model(level, graph, cache, fit, scales, basis, hydraulic_state, contract, seed):
    # Identical initialization across A/B/C except the conditioning source.
    torch.manual_seed(seed)
    np.random.seed(seed)
    reference = _build_reference(graph, cache, fit, scales)
    reference.load_state_dict(hydraulic_state, strict=True)
    prepared = prepare_static_v60(graph, "cpu")
    return DirectHydraulicEffectSurrogateV90(
        reference_model=reference,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=scales.state_delta_scale,
        flow_delta_scale=scales.flow_delta_scale,
        physics_dim=int(prepared.actuator_physics.shape[1]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_count=len(graph.actuator_ids),
        conditioning_level=level,
        contract=contract,
    )


def _preflight(model, cache, name, normalization, graph, device):
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    prepared = prepare_static_v80(graph, target)
    batch = cache.batch(name, normalization, target)
    kwargs = {}
    if model.conditioning_level == LEVEL_C:
        kwargs = {
            "oracle_reference_states_physical": batch.true_reference_states,
            "oracle_reference_flows_physical": batch.true_reference_flows,
        }
    output = model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        batch.reference_settings[:, None],
        batch.previous_actuator_flow,
        prepared,
        **kwargs,
    )
    if not torch.equal(output.raw_delta_states_physical, torch.zeros_like(output.raw_delta_states_physical)):
        raise RuntimeError("V9 exact-zero signed state effect failed")
    if not torch.equal(output.raw_delta_flows_physical, torch.zeros_like(output.raw_delta_flows_physical)):
        raise RuntimeError("V9 exact-zero signed flow effect failed")
    if any(p.requires_grad for p in model.reference_model.parameters()):
        raise RuntimeError("V9 modified frozen V7 Hydraulic parameters")
    return {"signed_state_exact_zero": True, "signed_flow_exact_zero": True, "reference_frozen": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 Step2 V9 D2 state-sufficiency ladder")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    started = time.perf_counter()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(f"V9 frozen actuator count requires 109, got {len(graph.actuator_ids)}")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V9 refuses legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    selected = sorted(d2 + d3)
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    if not fit_d2:
        raise ValueError("V9 ladder found no D2 TrainFit groups")
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V9 TrainFit/InternalHoldout leakage detected")

    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    value_checkpoint = _load_checkpoint(args.v70_value_checkpoint, "control_value")
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    contract = DirectHydraulicEffectLossContractV90()
    contract.validate()
    onset_positive_weight = derive_onset_sqrt_positive_weight_v80(
        cache, fit, normalization, contract=contract
    )
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    results: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    preflight: dict[str, Any] = {}
    for level in (LEVEL_A, LEVEL_B, LEVEL_C):
        model = _build_model(
            level, graph, cache, fit, scales, basis,
            hydraulic_checkpoint["state_dict"], contract, args.seed
        )
        preflight[level] = _preflight(model, cache, fit_d2[0], normalization, graph, str(target))
        histories[level] = train_d2_mechanism_v90(
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
        results[level] = evaluate_hydraulic_effect_v90(
            model,
            cache,
            fit_d2,
            normalization,
            prepared,
            scales,
            graph=graph,
            device=target,
            contract=contract,
        )

    decision = decide_state_sufficiency_v90(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_d2_digest = hashlib.sha256("\n".join(sorted(fit_d2)).encode("utf-8")).hexdigest()
    payload = {
        "contract": V90_CONTRACT,
        "development_only": True,
        "production_compatible": False,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "oracle_level_c_forbidden_online": True,
        "lineage": {
            "git_head": _git_head(),
            "implementation_sha256": _implementation_sha256(),
            "graph_sha256": _sha256(args.graph),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
            "hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
            "cache_lineage": cache_lineage,
            "fit_d2_group_count": len(fit_d2),
            "fit_d2_group_digest": fit_d2_digest,
            "seed": args.seed,
            "value_checkpoint_contract": value_checkpoint.get("contract"),
        },
        "diagnostic_schedule": {
            "source": "TRAINFIT_D2_ONLY",
            "epochs_per_level": int(contract.d2_pretrain_epochs),
            "optimizer": "AdamW",
            "learning_rate": float(contract.learning_rate),
            "weight_decay": float(contract.weight_decay),
            "grad_clip": float(contract.grad_clip),
            "fp32": True,
        },
        "preflight": preflight,
        "training_history": histories,
        "ladder": results,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path = out_dir / "step2_v90_state_sufficiency.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# PROJECT7 STEP2 V9 STATE SUFFICIENCY",
        "",
        f"- Contract: `{V90_CONTRACT}`",
        f"- Git HEAD: `{payload['lineage']['git_head']}`",
        f"- D2 TrainFit groups: `{len(fit_d2)}`; identical digest used for A/B/C: `{fit_d2_digest}`.",
        "- SWMM / Validation / Final / Formal: **not accessed**.",
        "- Level C uses authoritative true reference trajectory for oracle diagnosis only; forbidden online/production.",
        "",
        "## Decision",
        "",
        f"- `{decision['decision']}`",
        f"- Next: {decision['next_step']}",
        "",
    ]
    for level in (LEVEL_A, LEVEL_B, LEVEL_C):
        overall = results[level]["overall"]
        md += [
            f"## {level}",
            "",
            f"- depth skill: `{overall.get('delta_depth_m_skill_vs_zero', float('nan')):.4f}`",
            f"- flood skill: `{overall.get('delta_flood_m3s_skill_vs_zero', float('nan')):.4f}`",
            f"- storage skill: `{overall.get('delta_storage_m3_skill_vs_zero', float('nan')):.4f}`",
            f"- managed-flow skill: `{overall.get('delta_managed_flow_m3s_skill_vs_zero', float('nan')):.4f}`",
            "",
        ]
    (out_dir / "step2_v90_state_sufficiency.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"decision": decision, "report": str(json_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
