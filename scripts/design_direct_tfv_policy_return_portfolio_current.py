"""Design the current three-family same-prefix first-action portfolio from one causal context.

The pretrained Step2 stays 109-channel. Online proposal freedom is restricted by the native
supervisory-control artifact (82 facilities for the current Wuhan testbed), and q95 support is the
matching label-independent masked support rebuilt from existing D3 TrainFit actions. Current online
families are Step2 H10 scale 0.50, Step2 H10 scale 1.00 and type-aware hydraulic pressure. Projected
gradient remains code-level Development ablation only and is deliberately excluded here.

The emitted manifest is cryptographically bound to the exact causal context and its parent/asset
lineage. This prevents a scientifically valid candidate set from one query being reused at another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from rtc.direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from rtc.direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    changed_facility_support_limit,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph


PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_V4_EXACT_LINEAGE_82CONTROL_109REP"
)
CURRENT_THREE_FAMILY_SOURCES = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
)
_CONTEXT_TEXT_FIELDS = (
    "event_id",
    "rainfall_group",
    "recorded_prefix_action_sha256",
    "continuation_kind",
    "continuation_policy_sha256",
    "source_inp_sha256",
    "parent_decisions_sha256",
    "asset_manifest_sha256",
    "graph_sha256",
    "step2_checkpoint_sha256",
    "sequence_support_sha256",
    "supervisory_control_sha256",
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data:
        raise ValueError(f"policy-return context lacks lineage field {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"policy-return context field {key} must be scalar")
    return str(value.reshape(-1)[0])


def _scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    if key not in data:
        raise ValueError(f"policy-return context lacks lineage field {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"policy-return context field {key} must be scalar")
    return int(value.reshape(-1)[0])


def _joint_contract_h10_target(
    target: np.ndarray,
    active: np.ndarray,
    *,
    support: dict,
    quantile: str,
) -> tuple[np.ndarray, float, dict[str, float]]:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(active, dtype=np.float64)
    l1 = float(np.sum(np.abs(delta)))
    geometry = {
        "first_block_l1": l1,
        "h120_l1": l1,
        "h120_total_variation_l1": 2.0 * l1,
    }
    scale = 1.0
    for metric in SEQUENCE_SUPPORT_METRICS:
        mass = float(geometry[metric])
        limit = float(sequence_support_limit(support, metric, quantile))
        if mass > 1.0e-12:
            scale = min(scale, limit / mass)
    scale = float(np.clip(scale, 0.0, 1.0))
    contracted = (active + scale * delta).astype(np.float32)
    contracted_l1 = float(np.sum(np.abs(contracted.astype(np.float64) - active)))
    diagnostics = {
        "first_block_l1": contracted_l1,
        "h120_l1": contracted_l1,
        "h120_total_variation_l1": 2.0 * contracted_l1,
    }
    return contracted, scale, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--supervisory-control", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-chunk-size", type=int, default=24)
    # Deprecated launch-compatibility knobs. The current paper-facing designer does not use gradient.
    p.add_argument("--projected-gradient-steps", type=int, default=6)
    p.add_argument("--projected-gradient-step-fraction", type=float, default=0.25)
    args = p.parse_args()
    if int(args.projected_gradient_steps) <= 0:
        raise ValueError("projected-gradient-steps compatibility value must be positive")
    if not 0.0 < float(args.projected_gradient_step_fraction) <= 1.0:
        raise ValueError("projected-gradient-step-fraction compatibility value must lie in (0,1]")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("policy-return portfolio design requested CUDA but CUDA is unavailable")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.step2, graph=graph, device=device
    )
    action_support = dict(checkpoint["action_support"])
    first_radius = np.asarray(action_support["first_move_abs_q95_per_facility"], dtype=np.float32)
    control, mask = load_native_supervisory_control(
        args.supervisory_control,
        actuator_ids=graph.actuator_ids,
    )
    support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    step2_sha = _sha(args.step2)
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=step2_sha,
        supervisory_mask=mask,
        supervisory_control_contract=str(control["contract"]),
    )

    context_path = Path(args.context).resolve()
    if not context_path.is_file():
        raise FileNotFoundError(f"policy-return context does not exist: {context_path}")
    data = np.load(context_path, allow_pickle=False)
    if _scalar_text(data, "contract") != PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT:
        raise ValueError("policy-return context has the wrong exact-lineage contract")
    for key in ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow"):
        if key not in data:
            raise ValueError(f"policy-return context lacks {key}")
    context_lineage = {key: _scalar_text(data, key) for key in _CONTEXT_TEXT_FIELDS}
    context_lineage["decision_index"] = _scalar_int(data, "decision_index")
    context_lineage["decision_elapsed_seconds"] = _scalar_int(data, "decision_elapsed_seconds")
    expected_file_shas = {
        "graph_sha256": _sha(args.graph),
        "step2_checkpoint_sha256": step2_sha,
        "sequence_support_sha256": _sha(args.sequence_support),
        "supervisory_control_sha256": _sha(args.supervisory_control),
    }
    for key, expected in expected_file_shas.items():
        if str(context_lineage[key]).lower() != expected.lower():
            raise ValueError(f"policy-return context {key} differs from current designer input")

    state = torch.as_tensor(np.asarray(data["current_state"]), dtype=torch.float32, device=device)
    if state.ndim != 3 or int(state.shape[0]) != 1:
        raise ValueError("context current_state must be [1,node,state]")
    rain = torch.as_tensor(np.asarray(data["rainfall_scenarios"])[0], dtype=torch.float32, device=device)
    active = torch.as_tensor(np.asarray(data["active_target"])[0], dtype=torch.float32, device=device)
    flow = torch.as_tensor(np.asarray(data["previous_actuator_flow"]), dtype=torch.float32, device=device)
    ceiling = changed_facility_support_limit(support, "q95")

    hybrid = build_hybrid_policy_return_portfolio(
        model=model,
        normalization=normalization,
        graph=graph,
        current_state=state,
        rainfall_scenarios=rain,
        previous_actuator_flow=flow,
        active_target=active,
        first_radius=first_radius,
        max_changed_facilities=ceiling,
        max_delta_per_update=0.5,
        probe_chunk_size=int(args.probe_chunk_size),
        supervisory_mask=mask,
        include_projected_gradient_ablation=False,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_np = active.detach().cpu().numpy().astype(np.float64)
    rows = []
    seen: set[bytes] = set()
    for candidate in hybrid.candidates:
        if candidate.source not in CURRENT_THREE_FAMILY_SOURCES:
            raise RuntimeError(f"current designer received unexpected candidate family {candidate.source}")
        supported, contraction, geometry = _joint_contract_h10_target(
            candidate.target.detach().cpu().numpy(),
            active_np,
            support=support,
            quantile="q95",
        )
        if np.any(np.abs(supported.astype(np.float64)[~mask] - active_np[~mask]) > 1.0e-7):
            raise RuntimeError("final supported candidate changed a passive setting channel")
        changed = int(np.count_nonzero(np.abs(supported.astype(np.float64) - active_np) > 1.0e-7))
        if changed <= 0:
            continue
        key = np.ascontiguousarray(supported, dtype=np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        supported_tensor = torch.as_tensor(
            supported.reshape(1, 109), dtype=active.dtype, device=active.device
        )
        base_score = float(
            score_h10_first_action_targets(
                model=model,
                normalization=normalization,
                graph=graph,
                current_state=state,
                rainfall_scenarios=rain,
                previous_actuator_flow=flow,
                active_target=active,
                candidate_targets=supported_tensor,
                probe_chunk_size=1,
            )[0].detach().cpu()
        )
        payload = {
            "contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            "candidate_source": candidate.source,
            "actuator_ids": [str(value) for value in graph.actuator_ids],
            "target_settings": supported.astype(float).tolist(),
            "changed_facility_count": changed,
            "base_step2_h10_score_m3": base_score,
            "active_support_ceiling": ceiling,
            "supervisory_control_dimension": int(mask.sum()),
            "model_action_channel_count": 109,
            "supervisory_mask_sha256": str(control["supervisory_mask_sha256"]),
            "passive_setting_channels_unchanged": True,
            "joint_sequence_support_quantile": "q95",
            "joint_sequence_radial_contraction": contraction,
            "joint_sequence_geometry": geometry,
            "action_token_semantics": "H10_CANDIDATE_THEN_H350_HOLD",
            "future_realized_rainfall_used": False,
            "online_swmm_called": False,
        }
        path = out / f"candidate_{len(rows):02d}_{candidate.source.lower()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"path": str(path.resolve()), **payload})
    if not rows:
        raise RuntimeError("three-family policy-return portfolio produced no distinct non-HOLD candidate")
    if len(rows) > 3:
        raise RuntimeError("current policy-return portfolio exceeded the three-candidate contract")

    context_sha = _sha(context_path)
    manifest = {
        "contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "h10_probe_generator_contract": DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
        "candidate_count": len(rows),
        "candidate_family_count_max": 3,
        "candidate_family_contract": list(CURRENT_THREE_FAMILY_SOURCES),
        "active_support_ceiling": ceiling,
        "supervisory_control_dimension": int(mask.sum()),
        "model_action_channel_count": 109,
        "supervisory_control_contract": str(control["contract"]),
        "supervisory_mask_sha256": str(control["supervisory_mask_sha256"]),
        "probe_count": int(hybrid.learned_probe.probe_count),
        "predicted_beneficial_facility_count": int(hybrid.learned_probe.predicted_beneficial_facility_count),
        "selected_probe_facility_indices": list(hybrid.learned_probe.selected_facility_indices),
        "candidate_sources": [row["candidate_source"] for row in rows],
        "candidates": rows,
        "context_npz": str(context_path),
        "context_npz_sha256": context_sha,
        "context_contract": PRACTICAL_RTC_CAUSAL_QUERY_CONTEXT_CONTRACT,
        **context_lineage,
        "projected_gradient_online": False,
        "projected_gradient_ablation_available": True,
        "projected_gradient_generator_contract": DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
        "projected_gradient_role": "HISTORICAL_OR_EXPLICIT_DEVELOPMENT_ABLATION_ONLY",
        "lbfgsb_used": False,
        "future_realized_rainfall_used": False,
        "online_swmm_called": False,
    }
    manifest_path = out / "POLICY_RETURN_CANDIDATE_PORTFOLIO.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
