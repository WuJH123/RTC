"""Design the Development calibration panel for the executable H10 receding-prefix estimand.

This script never runs SWMM. It reuses the already-frozen V6 raw-optimizer policy panel and replaces
each full H120/H360 candidate with the action actually committed before replanning: its first 10-minute
block followed by HOLD for the remaining H350. One HOLD and one prefix candidate are emitted per
fresh admission rainfall group for authoritative offline SWMM labeling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.data_design import canonical_sequence_sha
from rtc.direct_tfv_policy_admission import (
    DIRECT_TFV_POLICY_PANEL_CONTRACT,
    DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
)
from rtc.direct_tfv_receding_prefix import (
    DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
    executable_prefix_sequence,
)
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d3_design_v60 import D3_V60_HOLD_ROLE
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache


PREFIX_CANDIDATE_ROLE = "D3_V8_RECEDING_PREFIX_CALIBRATION_CANDIDATE"
CURRENT_PREFIX_PANEL_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_RECEDING_PREFIX_PANEL_DESIGN_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _physical_inputs(batch, normalization):
    sm = torch.as_tensor(
        normalization.state_mean, dtype=batch.initial_state.dtype, device=batch.initial_state.device
    )
    ss = torch.as_tensor(
        normalization.state_std, dtype=batch.initial_state.dtype, device=batch.initial_state.device
    )
    rm = torch.as_tensor(
        normalization.rainfall_mean, dtype=batch.rainfall.dtype, device=batch.rainfall.device
    )
    rs = torch.as_tensor(
        normalization.rainfall_std, dtype=batch.rainfall.dtype, device=batch.rainfall.device
    )
    fm = torch.as_tensor(
        normalization.flow_mean,
        dtype=batch.previous_actuator_flow.dtype,
        device=batch.previous_actuator_flow.device,
    )
    fs = torch.as_tensor(
        normalization.flow_std,
        dtype=batch.previous_actuator_flow.dtype,
        device=batch.previous_actuator_flow.device,
    )
    return (
        batch.initial_state * ss + sm,
        batch.rainfall * rs + rm,
        batch.previous_actuator_flow * fs + fm,
    )


def _score_sequence(
    model, normalization, graph, *, state, rainfall, flow, active_target, sequence
):
    sm = torch.as_tensor(normalization.state_mean, dtype=state.dtype, device=state.device)
    ss = torch.as_tensor(normalization.state_std, dtype=state.dtype, device=state.device).clamp_min(
        1.0e-6
    )
    rm = torch.as_tensor(normalization.rainfall_mean, dtype=rainfall.dtype, device=rainfall.device)
    rs = torch.as_tensor(
        normalization.rainfall_std, dtype=rainfall.dtype, device=rainfall.device
    ).clamp_min(1.0e-6)
    fm = torch.as_tensor(normalization.flow_mean, dtype=flow.dtype, device=flow.device)
    fs = torch.as_tensor(normalization.flow_std, dtype=flow.dtype, device=flow.device).clamp_min(1.0e-6)
    normalized_state = (state - sm) / ss
    normalized_rain = (rainfall - rm) / rs
    normalized_flow = (flow - fm) / fs
    reference = active_target[None].expand(72, -1)[None]
    with torch.no_grad():
        output = model(
            current_state=normalized_state,
            rainfall=normalized_rain,
            reference_settings=reference,
            candidate_settings=sequence[None],
            previous_actuator_flow=normalized_flow,
            actuator_upstream=torch.as_tensor(
                graph.actuator_upstream, dtype=torch.long, device=state.device
            ),
            actuator_downstream=torch.as_tensor(
                graph.actuator_downstream, dtype=torch.long, device=state.device
            ),
            actuator_physics=torch.as_tensor(
                graph.actuator_physics, dtype=state.dtype, device=state.device
            ),
        )
    return float(output.total_delta_tfv_m3[0].detach().cpu())


def _sequence(ids: tuple[str, ...], blocks: np.ndarray) -> list[dict[str, float]]:
    return [{aid: float(step[i]) for i, aid in enumerate(ids)} for step in blocks]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--fresh-cache-manifest", required=True)
    p.add_argument("--fresh-causal-store", required=True)
    p.add_argument("--fresh-causal-state-store", required=True)
    p.add_argument("--policy-design-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    graph = _load_graph(args.graph)
    model, normalization, _ = load_direct_tfv_runtime_checkpoint(
        args.checkpoint, graph=graph, device=device
    )
    fresh = V60TrainCache(args.fresh_cache_manifest)
    rain = load_causal_forecast_store_v123(args.fresh_causal_store)
    state = load_causal_state_store_v127(args.fresh_causal_state_store)
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(fresh, rain), state)
    policy = pd.read_csv(args.policy_design_manifest)
    required = {
        "rainfall_group",
        "event_id",
        "checkpoint_id",
        "data_role",
        "settings_sequence_json",
        "sequence_sha256",
        "policy_panel_contract",
        "policy_query_step3_contract",
        "predicted_delta_tfv_m3",
        "active_facility_count",
    }
    missing = sorted(required - set(policy.columns))
    if missing:
        raise ValueError(f"policy design manifest missing columns: {missing}")
    candidates = policy[
        policy["data_role"].astype(str) == "D3_V6_POLICY_CALIBRATION_CANDIDATE"
    ].copy()
    holds = policy[policy["data_role"].astype(str).str.upper() == D3_V60_HOLD_ROLE].copy()
    if candidates.empty or len(candidates) != len(holds):
        raise ValueError("policy design must contain one V6 candidate and one HOLD per group")
    if set(candidates["policy_panel_contract"].astype(str)) != {
        DIRECT_TFV_POLICY_PANEL_CONTRACT
    }:
        raise ValueError("input policy manifest has the wrong panel contract")
    if set(candidates["policy_query_step3_contract"].astype(str)) != {
        DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT
    }:
        raise ValueError("input policy manifest was not generated by V6 raw optimizer")

    actuator_ids = tuple(str(x) for x in graph.actuator_ids)
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    seen: set[str] = set()
    by_checkpoint = {str(row.checkpoint_id): row for row in candidates.itertuples(index=False)}
    hold_by_checkpoint = {str(row.checkpoint_id): row for row in holds.itertuples(index=False)}

    for name in sorted(fresh.targeted_d3_names()):
        entry = fresh.entry(name)
        checkpoint = str(entry.checkpoint_id)
        candidate_row = by_checkpoint.get(checkpoint)
        hold_row = hold_by_checkpoint.get(checkpoint)
        if candidate_row is None or hold_row is None:
            raise ValueError(f"{name}: policy design lacks matching candidate/HOLD")
        group = str(entry.rainfall_group)
        if group in seen:
            raise ValueError(f"duplicate fresh rainfall group in prefix panel: {group}")
        seen.add(group)
        if str(candidate_row.rainfall_group) != group or str(candidate_row.event_id) != str(
            entry.event_id
        ):
            raise ValueError(f"{name}: policy/fresh identity mismatch")

        batch = online.batch(name, normalization, device)
        active_target = batch.reference_settings[0, 0]
        state_raw, rain_raw, flow_raw = _physical_inputs(batch, normalization)
        full_blocks = np.asarray(
            [
                [float(step[aid]) for aid in actuator_ids]
                for step in json.loads(str(candidate_row.settings_sequence_json))
            ],
            dtype=np.float32,
        )
        if full_blocks.shape != (36, 109):
            raise ValueError(f"{name}: policy candidate is not [36,109]")
        full_tensor = torch.as_tensor(
            full_blocks, dtype=state_raw.dtype, device=device
        ).repeat_interleave(2, dim=0)
        prefix_tensor = executable_prefix_sequence(
            full_tensor, active_target, control_block_steps=2
        )
        prefix_blocks = (
            prefix_tensor.reshape(36, 2, 109)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        prefix_sequence = _sequence(actuator_ids, prefix_blocks)
        prefix_sha = canonical_sequence_sha(prefix_sequence)
        prefix_score = _score_sequence(
            model,
            normalization,
            graph,
            state=state_raw,
            rainfall=rain_raw,
            flow=flow_raw,
            active_target=active_target,
            sequence=prefix_tensor,
        )
        changed = int(
            torch.sum(torch.abs(prefix_tensor[0] - active_target) > 1.0e-7).detach().cpu()
        )
        if changed <= 0:
            raise ValueError(f"{name}: V6 raw optimizer produced no executable first move")

        hold_output = hold_row._asdict()
        hold_output.update(
            {
                "receding_prefix_panel_contract": DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
                "receding_prefix_query_step3_contract": DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
                "receding_prefix_role": "HOLD_REFERENCE",
            }
        )
        rows.append(hold_output)

        prefix_output = hold_row._asdict()
        prefix_output.update(
            {
                "data_role": PREFIX_CANDIDATE_ROLE,
                "sequence_index": 1,
                "candidate_family": "v8_execute_h10_then_hold_h350",
                "v60_coefficients_json": json.dumps([]),
                "settings_sequence_json": json.dumps(prefix_sequence, sort_keys=True),
                "sequence_sha256": prefix_sha,
                "active_control_groups": -1,
                "active_actuators": changed,
                "sequence_rate_feasible": True,
                "receding_prefix_panel_contract": DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
                "receding_prefix_query_step3_contract": DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
                "receding_prefix_role": "EXECUTABLE_PREFIX_QUERY",
                "receding_prefix_semantics": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
                "predicted_prefix_delta_tfv_m3": float(prefix_score),
                "full_plan_predicted_delta_tfv_m3": float(candidate_row.predicted_delta_tfv_m3),
                "full_plan_sequence_sha256": str(candidate_row.sequence_sha256),
                "active_facility_count": int(candidate_row.active_facility_count),
                "prefix_changed_facility_count": changed,
            }
        )
        rows.append(prefix_output)
        summary.append(
            {
                "group": name,
                "rainfall_group": group,
                "event_id": str(entry.event_id),
                "checkpoint_id": checkpoint,
                "prefix_sequence_sha256": prefix_sha,
                "full_plan_sequence_sha256": str(candidate_row.sequence_sha256),
                "predicted_prefix_delta_tfv_m3": float(prefix_score),
                "full_plan_predicted_delta_tfv_m3": float(candidate_row.predicted_delta_tfv_m3),
                "active_facility_count": int(candidate_row.active_facility_count),
                "prefix_changed_facility_count": changed,
            }
        )

    frame = pd.DataFrame.from_records(rows)
    if len(frame) != 2 * len(seen) or len(seen) < 9:
        raise RuntimeError(
            "receding-prefix panel requires one HOLD and one prefix candidate per rainfall group"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    payload = {
        "contract": CURRENT_PREFIX_PANEL_RUN_CONTRACT,
        "development_only": True,
        "receding_prefix_panel_contract": DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
        "policy_query_step3_contract": DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
        "execution_estimand": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
        "density_classification_variable": "FIRST_MOVE_CHANGED_FACILITY_COUNT",
        "rainfall_group_count": len(seen),
        "rows": len(frame),
        "hold_rows": len(seen),
        "candidate_rows": len(seen),
        "records": summary,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "fresh_cache_sha256": _sha(args.fresh_cache_manifest),
            "fresh_causal_rainfall_sha256": _sha(args.fresh_causal_store),
            "fresh_causal_state_sha256": _sha(args.fresh_causal_state_store),
            "policy_design_manifest_sha256": _sha(args.policy_design_manifest),
        },
        "online_swmm_called": False,
    }
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
