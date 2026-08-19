"""Design the V14 causal first-target portfolio from one captured policy-return context.

This performs no SWMM simulation and reads no future realised rainfall.  It converts one causal
state/rainfall context plus its learned parent target into supported target JSON files that can be
fed to ``run_direct_tfv_policy_return_pair_current.py --candidate-target-json``.  The final radial
contraction uses the same q95 D3-HOLD joint-sequence metrics as online Direct-TFV control.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_policy_return_candidate_portfolio,
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


def _joint_contract_target(
    target: np.ndarray,
    active: np.ndarray,
    *,
    support: dict,
    quantile: str,
    free_control_blocks: int = 12,
) -> tuple[np.ndarray, float]:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(active, dtype=np.float64)
    l1 = float(np.sum(np.abs(delta)))
    geometry = {
        "first_block_l1": l1,
        "h120_l1": float(free_control_blocks) * l1,
        "h120_total_variation_l1": l1,
    }
    scale = 1.0
    for metric in SEQUENCE_SUPPORT_METRICS:
        mass = float(geometry[metric])
        limit = float(sequence_support_limit(support, metric, quantile))
        if mass > 1.0e-12:
            scale = min(scale, limit / mass)
    scale = float(np.clip(scale, 0.0, 1.0))
    return (active + scale * delta).astype(np.float32), scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    graph = _load_graph(args.graph)
    _, _, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.step2, graph=graph, device=torch.device("cpu")
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
    for key in (
        "current_state",
        "rainfall_scenarios",
        "active_target",
        "candidate_target",
    ):
        if key not in data:
            raise ValueError(f"policy-return context lacks {key}")
    state = torch.as_tensor(np.asarray(data["current_state"]), dtype=torch.float32)
    if state.ndim == 3 and state.shape[0] == 1:
        pass
    else:
        raise ValueError("context current_state must be [1,node,state]")
    rain = torch.as_tensor(np.asarray(data["rainfall_scenarios"])[0], dtype=torch.float32)
    active = torch.as_tensor(np.asarray(data["active_target"])[0], dtype=torch.float32)
    learned = torch.as_tensor(np.asarray(data["candidate_target"])[0], dtype=torch.float32)
    ceiling = _active_ceiling(action_support)
    candidates = build_policy_return_candidate_portfolio(
        current_state=state,
        rainfall_scenarios=rain,
        active_target=active,
        v12_target=learned,
        graph=graph,
        first_radius=first_radius,
        max_changed_facilities=ceiling,
        max_delta_per_update=0.5,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_np = active.detach().cpu().numpy().astype(np.float64)
    rows = []
    seen: set[bytes] = set()
    for index, candidate in enumerate(candidates):
        supported, contraction = _joint_contract_target(
            candidate.target.detach().cpu().numpy(),
            active_np,
            support=support,
            quantile="q95",
        )
        changed = int(np.count_nonzero(np.abs(supported.astype(np.float64) - active_np) > 1.0e-7))
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
            "future_realized_rainfall_used": False,
            "online_swmm_called": False,
        }
        path = out / f"candidate_{len(rows):02d}_{candidate.source.lower()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"path": str(path.resolve()), **payload})
    if len(rows) < 2:
        raise RuntimeError("policy-return portfolio produced fewer than two distinct supported candidates")
    manifest = {
        "contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "candidate_count": len(rows),
        "active_support_ceiling": ceiling,
        "candidate_sources": [row["candidate_source"] for row in rows],
        "candidates": rows,
    }
    manifest_path = out / "POLICY_RETURN_CANDIDATE_PORTFOLIO.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
