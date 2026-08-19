"""Design the current same-prefix first-action portfolio from one captured causal context.

No SWMM simulation and no future realised rainfall are used. Frozen base Step2 provides finite H10
probes plus one differentiable 109-D projected-gradient proposal. Every target is first-move/q95
supported, then contracted to the actual H10-pulse joint-sequence q95 geometry.
"""
from __future__ import annotations

import argparse
import json
import math
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
from rtc.direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from rtc.production_cli import _load_graph
from rtc.step2_tfv_support import DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT


def _active_ceiling(action_support: dict) -> int:
    extension = str(action_support.get("joint_density_extension_contract", ""))
    key = "joint_changed_facility_count_q95"
    if extension != DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT or key not in action_support:
        key = "joint_changed_facility_count_q90"
    value = float(action_support.get(key, 1.0))
    observed = int(action_support.get("joint_changed_facility_count_max", max(1, math.ceil(value))))
    return max(1, min(109, observed, int(math.ceil(value))))


def _joint_contract_h10_target(
    target: np.ndarray,
    active: np.ndarray,
    *,
    support: dict,
    quantile: str,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Contract the actual H10 pulse geometry to the frozen joint-sequence trust region."""
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
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-chunk-size", type=int, default=24)
    p.add_argument("--projected-gradient-steps", type=int, default=6)
    p.add_argument("--projected-gradient-step-fraction", type=float, default=0.25)
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.step2, graph=graph, device=device
    )
    action_support = dict(checkpoint["action_support"])
    first_radius = np.asarray(action_support["first_move_abs_q95_per_facility"], dtype=np.float32)
    support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=None,
    )

    data = np.load(args.context, allow_pickle=False)
    for key in ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow"):
        if key not in data:
            raise ValueError(f"policy-return context lacks {key}")
    state = torch.as_tensor(np.asarray(data["current_state"]), dtype=torch.float32, device=device)
    if state.ndim != 3 or int(state.shape[0]) != 1:
        raise ValueError("context current_state must be [1,node,state]")
    rain = torch.as_tensor(
        np.asarray(data["rainfall_scenarios"])[0], dtype=torch.float32, device=device
    )
    active = torch.as_tensor(
        np.asarray(data["active_target"])[0], dtype=torch.float32, device=device
    )
    flow = torch.as_tensor(
        np.asarray(data["previous_actuator_flow"]), dtype=torch.float32, device=device
    )
    ceiling = _active_ceiling(action_support)

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
        gradient_steps=int(args.projected_gradient_steps),
        gradient_step_fraction=float(args.projected_gradient_step_fraction),
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_np = active.detach().cpu().numpy().astype(np.float64)
    rows = []
    seen: set[bytes] = set()
    for candidate in hybrid.candidates:
        supported, contraction, geometry = _joint_contract_h10_target(
            candidate.target.detach().cpu().numpy(),
            active_np,
            support=support,
            quantile="q95",
        )
        changed = int(
            np.count_nonzero(np.abs(supported.astype(np.float64) - active_np) > 1.0e-7)
        )
        if changed <= 0:
            continue
        key = np.ascontiguousarray(supported, dtype=np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        payload = {
            "contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            "candidate_source": candidate.source,
            "actuator_ids": [str(value) for value in graph.actuator_ids],
            "target_settings": supported.astype(float).tolist(),
            "changed_facility_count": changed,
            "active_support_ceiling": ceiling,
            "joint_sequence_support_quantile": "q95",
            "joint_sequence_radial_contraction": contraction,
            "joint_sequence_geometry": geometry,
            "action_token_semantics": "H10_CANDIDATE_THEN_H350_HOLD",
            "future_realized_rainfall_used": False,
            "online_swmm_called": False,
        }
        path = out / f"candidate_{len(rows):02d}_{candidate.source.lower()}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append({"path": str(path.resolve()), **payload})
    if len(rows) < 2:
        raise RuntimeError("hybrid policy-return portfolio produced fewer than two distinct candidates")
    if len(rows) > 4:
        raise RuntimeError("hybrid policy-return portfolio exceeded the four-candidate contract")

    gradient = hybrid.projected_gradient
    manifest = {
        "contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "h10_probe_generator_contract": DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
        "projected_gradient_generator_contract": DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
        "candidate_count": len(rows),
        "candidate_family_count_max": 4,
        "active_support_ceiling": ceiling,
        "probe_count": int(hybrid.learned_probe.probe_count),
        "predicted_beneficial_facility_count": int(
            hybrid.learned_probe.predicted_beneficial_facility_count
        ),
        "selected_probe_facility_indices": list(hybrid.learned_probe.selected_facility_indices),
        "candidate_sources": [row["candidate_source"] for row in rows],
        "candidates": rows,
        "projected_gradient": {
            "source": "SUPPORT_CONSTRAINED_GRADIENT_H10",
            "produced_nonhold_candidate": bool(gradient.produced_nonhold_candidate),
            "attempted_steps": int(gradient.attempted_steps),
            "accepted_improvement_steps": int(gradient.accepted_improvement_steps),
            "start_score_m3": float(gradient.start_score_m3),
            "best_score_m3": float(gradient.best_score_m3),
            "final_gradient_l2": float(gradient.final_gradient_l2),
        },
        "lbfgsb_used": False,
        "gradient_dimension": 109,
        "gradient_action_horizon": "H10_ONLY",
        "future_realized_rainfall_used": False,
        "online_swmm_called": False,
    }
    manifest_path = out / "POLICY_RETURN_CANDIDATE_PORTFOLIO.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
