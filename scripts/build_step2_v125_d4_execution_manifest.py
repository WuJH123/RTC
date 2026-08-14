"""Convert the frozen V125 D4 V2 plan into the existing guarded D3 SWMM manifest.

This script does not run SWMM and does not redesign any candidate.  It is a strict
adapter between the 5-minute Step2 scoring representation and the already validated
`rtc-run-d3-batch` 10-minute control-sequence runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.data_design import canonical_sequence_sha
from rtc.production_cli import _load_graph
from rtc.step2_d3_design_v60 import D3_FEASIBILITY_CONTRACT, D3_TIME_CONTRACT
from rtc.step2_d4_action_support_v125 import D4_ACTION_SUPPORT_CONTRACT_V125

EXECUTION_MANIFEST_CONTRACT = "PROJECT7_V125_D4_GUARDED_SWMM_EXECUTION_MANIFEST_V1"
D4_ANCHOR_ROLE = "D4_V125_ANCHOR_REFERENCE"
D4_CANDIDATE_ROLE = "D4_V125_ANCHOR_NEIGHBOURHOOD_CANDIDATE"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _join_metadata(plan: pd.DataFrame, checkpoints: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "rainfall_group", "checkpoint_id", "inp_path", "trajectory_metadata_path"}
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise ValueError(f"V125 checkpoint metadata missing columns: {missing}")
    meta = checkpoints.copy()
    if "scientific_split" in meta.columns:
        meta = meta[meta["scientific_split"].astype(str).str.lower() == "development"]
    if "development_fold" in meta.columns:
        meta = meta[meta["development_fold"].astype(str).str.lower() == "train"]
    keys = ["event_id", "rainfall_group", "checkpoint_id"]
    if meta.duplicated(keys, keep=False).any():
        raise ValueError("V125 checkpoint execution metadata is not one-to-one")
    keep = keys + ["inp_path", "trajectory_metadata_path"]
    merged = plan.merge(meta[keep], how="left", on=keys, validate="many_to_one")
    if merged[["inp_path", "trajectory_metadata_path"]].isna().any().any():
        bad = merged.loc[
            merged["inp_path"].isna() | merged["trajectory_metadata_path"].isna(), keys
        ].drop_duplicates()
        raise ValueError(f"V125 D4 plan cannot resolve execution metadata for {bad.to_dict('records')[:5]}")
    return merged


def _runner_blocks(sequence_json: str, actuator_ids: tuple[str, ...]) -> tuple[list[dict[str, float]], np.ndarray]:
    model_sequence = np.asarray(json.loads(str(sequence_json)), dtype=np.float32)
    if model_sequence.ndim != 2 or model_sequence.shape[0] != 72:
        raise ValueError("V125 D4 scoring sequence must be H72 [72,actuator]")
    if model_sequence.shape[1] != len(actuator_ids) or not np.isfinite(model_sequence).all():
        raise ValueError("V125 D4 scoring sequence actuator identity/finite check failed")
    paired = model_sequence.reshape(36, 2, len(actuator_ids))
    pair_error = float(np.max(np.abs(paired[:, 0] - paired[:, 1])))
    if pair_error > 1.0e-7:
        raise ValueError(
            "V125 D4 plan is not executable at 10-minute cadence: paired 5-minute targets differ "
            f"(max={pair_error})"
        )
    blocks = paired[:, 0, :].copy()
    settings = [
        {aid: float(block[i]) for i, aid in enumerate(actuator_ids)}
        for block in blocks
    ]
    return settings, blocks


def build_execution_manifest(
    *,
    plan_path: str | Path,
    checkpoint_metadata_path: str | Path,
    graph_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    plan = pd.read_csv(plan_path)
    required = {
        "contract",
        "plan_row_id",
        "split_role",
        "event_id",
        "rainfall_group",
        "checkpoint_id",
        "elapsed_seconds",
        "candidate_family",
        "action_sequence_json",
        "action_sequence_sha256",
        "control_block_steps",
        "horizon_steps",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"V125 D4 plan missing columns: {missing}")
    if set(plan["contract"].astype(str)) != {D4_ACTION_SUPPORT_CONTRACT_V125}:
        raise ValueError("V125 D4 plan contract is stale; rerun the V2 common-continuation planner")
    if set(plan["split_role"].astype(str)) - {"fit", "audit"}:
        raise ValueError("V125 D4 plan contains invalid fit/audit roles")
    split_by_rain = plan.groupby("rainfall_group")["split_role"].nunique()
    if bool((split_by_rain != 1).any()):
        raise ValueError("V125 D4 fit/audit split leaks within a rainfall group")
    graph = _load_graph(graph_path)
    actuator_ids = tuple(str(x) for x in graph.actuator_ids)
    if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
        raise ValueError("V125 D4 execution requires the frozen 109-actuator identity")
    checkpoints = pd.read_csv(checkpoint_metadata_path)
    frame = _join_metadata(plan, checkpoints)

    records: list[dict[str, object]] = []
    common_tail_by_checkpoint: dict[str, np.ndarray] = {}
    first_move_by_checkpoint: dict[str, set[bytes]] = {}
    for _, row in frame.iterrows():
        settings, blocks = _runner_blocks(str(row["action_sequence_json"]), actuator_ids)
        checkpoint = str(row["checkpoint_id"])
        tail = blocks[1:].copy()
        old_tail = common_tail_by_checkpoint.setdefault(checkpoint, tail)
        if not np.array_equal(old_tail, tail):
            raise RuntimeError("V125 D4 common-continuation invariant failed within checkpoint")
        first_key = np.round(blocks[0], 7).astype("<f4", copy=False).tobytes()
        seen = first_move_by_checkpoint.setdefault(checkpoint, set())
        if first_key in seen:
            raise RuntimeError("V125 D4 execution manifest collapsed first moves within checkpoint")
        seen.add(first_key)
        max_step = float(np.max(np.abs(np.diff(blocks, axis=0)))) if len(blocks) > 1 else 0.0
        if max_step > 0.5000001:
            raise ValueError(f"V125 D4 sequence violates 0.5 rate limit: {max_step}")
        elapsed = int(row["elapsed_seconds"])
        if elapsed < 0 or elapsed % 60:
            raise ValueError("V125 D4 checkpoint elapsed time must be whole minutes")
        family = str(row["candidate_family"])
        role = D4_ANCHOR_ROLE if family == "anchor_scale_1.00" else D4_CANDIDATE_ROLE
        runner_sha = canonical_sequence_sha(settings)
        records.append({
            "v125_execution_manifest_contract": EXECUTION_MANIFEST_CONTRACT,
            "v125_d4_contract": D4_ACTION_SUPPORT_CONTRACT_V125,
            "plan_row_id": str(row["plan_row_id"]),
            "d4_split_role": str(row["split_role"]),
            "event_id": str(row["event_id"]),
            "rainfall_group": str(row["rainfall_group"]),
            "scientific_split": "development",
            "development_fold": "train",
            "checkpoint_id": checkpoint,
            "checkpoint_minutes": elapsed // 60,
            "inp_path": str(row["inp_path"]),
            "trajectory_metadata_path": str(row["trajectory_metadata_path"]),
            "data_role": role,
            "candidate_family": family,
            "sequence_index": int(row.get("selected_rank", 0)),
            "settings_sequence_json": json.dumps(settings, sort_keys=True, separators=(",", ":")),
            "sequence_sha256": runner_sha,
            "d4_scoring_sequence_sha256": str(row["action_sequence_sha256"]),
            "model_horizon_steps": 72,
            "model_step_seconds": 300,
            "control_update_seconds": 600,
            "control_block_steps": 2,
            "control_blocks": 36,
            "d3_time_contract": D3_TIME_CONTRACT,
            "d3_feasibility_contract": D3_FEASIBILITY_CONTRACT,
            "sequence_rate_feasible": True,
            "all_actuators_eligible": True,
            "fixed_active_subset": False,
            "future_action_rule": "candidate_first_600s_then_exact_common_sparse_rbc_anchor_continuation",
        })

    out = pd.DataFrame.from_records(records)
    if out.duplicated(["checkpoint_id", "sequence_sha256"]).any():
        raise RuntimeError("V125 D4 execution manifest contains duplicate checkpoint sequences")
    anchor_counts = out[out["data_role"] == D4_ANCHOR_ROLE].groupby("checkpoint_id").size()
    if len(anchor_counts) != out["checkpoint_id"].nunique() or not (anchor_counts == 1).all():
        raise RuntimeError("V125 D4 requires exactly one anchor reference per checkpoint")
    fit_rain = set(out.loc[out["d4_split_role"] == "fit", "rainfall_group"].astype(str))
    audit_rain = set(out.loc[out["d4_split_role"] == "audit", "rainfall_group"].astype(str))
    if fit_rain & audit_rain:
        raise RuntimeError("V125 D4 execution manifest has fit/audit rainfall leakage")
    summary: dict[str, object] = {
        "contract": EXECUTION_MANIFEST_CONTRACT,
        "rows": int(len(out)),
        "checkpoints": int(out["checkpoint_id"].nunique()),
        "actuators": len(actuator_ids),
        "control_blocks": 36,
        "control_update_seconds": 600,
        "fit_rainfall_groups": sorted(fit_rain),
        "audit_rainfall_groups": sorted(audit_rain),
        "rainfall_overlap": sorted(fit_rain & audit_rain),
        "anchor_rows": int((out["data_role"] == D4_ANCHOR_ROLE).sum()),
        "candidate_rows": int((out["data_role"] == D4_CANDIDATE_ROLE).sum()),
        "plan_sha256": _sha(plan_path),
        "checkpoint_metadata_sha256": _sha(checkpoint_metadata_path),
        "graph_sha256": _sha(graph_path),
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    return out, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args()
    frame, summary = build_execution_manifest(
        plan_path=args.plan,
        checkpoint_metadata_path=args.checkpoints,
        graph_path=args.graph,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary["out"] = str(out.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
