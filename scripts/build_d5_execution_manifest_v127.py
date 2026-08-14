"""Convert the frozen V127 D5 plan to the validated rtc-run-d3-batch manifest.

No SWMM is run and no D5 action is redesigned.  The adapter verifies exact 10-minute
target-latch executability, the first move relative to the causal active target, central
pair identities, direct 12x109 MPC-variable lineage and checkpoint lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.d5_gradient_v127 import V127_D5_CONTRACT, sequence_sha256_v127
from rtc.data_design import canonical_sequence_sha
from rtc.production_cli import _load_graph
from rtc.step2_d3_design_v60 import D3_FEASIBILITY_CONTRACT, D3_TIME_CONTRACT

V127_D5_EXECUTION_CONTRACT = "PROJECT7_V127_D5_GUARDED_SWMM_EXECUTION_MANIFEST_V2_1308VAR"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(raw: str, count: int, *, label: str) -> np.ndarray:
    value = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError(f"V127 D5 {label} must be a finite actuator vector")
    return value


def _fractions(raw: str, actuator_count: int, *, label: str) -> np.ndarray:
    value = np.asarray(json.loads(raw), dtype=np.float32)
    if value.shape != (12, actuator_count) or not np.isfinite(value).all():
        raise ValueError(f"V127 D5 {label} must be finite 12 x actuator_count")
    if np.any((value < -1e-8) | (value > 1.0 + 1e-8)):
        raise ValueError(f"V127 D5 {label} leaves L-BFGS-B [0,1] bounds")
    return value


def _blocks(
    raw: str,
    actuator_ids: tuple[str, ...],
    *,
    active_target: np.ndarray,
) -> tuple[list[dict[str, float]], np.ndarray]:
    sequence = np.asarray(json.loads(raw), dtype=np.float32)
    if sequence.shape != (72, len(actuator_ids)) or not np.isfinite(sequence).all():
        raise ValueError("V127 D5 action sequence must be finite H72 x actuator_count")
    paired = sequence.reshape(36, 2, len(actuator_ids))
    if float(np.max(np.abs(paired[:, 0] - paired[:, 1]))) > 1.0e-7:
        raise ValueError("V127 D5 sequence changes inside a 10-min control block")
    block = paired[:, 0].copy()
    first_delta = float(np.max(np.abs(block[0] - active_target)))
    later_delta = float(np.max(np.abs(np.diff(block, axis=0)))) if len(block) > 1 else 0.0
    if first_delta > 0.5000001 or later_delta > 0.5000001:
        raise ValueError(
            f"V127 D5 sequence violates max 0.5 target change: first={first_delta}, later={later_delta}"
        )
    if float(np.max(np.abs(block[12:] - block[11][None, :]))) > 1.0e-7:
        raise ValueError("V127 D5 sequence does not hold terminal H120 target through H360")
    settings = [
        {aid: float(values[i]) for i, aid in enumerate(actuator_ids)}
        for values in block
    ]
    return settings, block


def build_manifest(
    *, plan_path: str | Path, checkpoint_metadata_path: str | Path, graph_path: str | Path
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
        "center_id",
        "center_family",
        "probe_role",
        "direction_id",
        "direction_family",
        "direction_fractions_json",
        "center_fractions_json",
        "probe_fractions_json",
        "active_target_json",
        "direct_mpc_variable_count",
        "epsilon",
        "action_sequence_sha256",
        "action_sequence_json",
        "free_control_blocks",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"V127 D5 plan missing columns: {missing}")
    if set(plan["contract"].astype(str)) != {V127_D5_CONTRACT}:
        raise ValueError("V127 D5 plan contract mismatch")
    if set(plan["probe_role"].astype(str)) - {"center", "plus", "minus"}:
        raise ValueError("V127 D5 plan contains invalid probe roles")
    if set(plan["split_role"].astype(str)) - {"fit", "audit"}:
        raise ValueError("V127 D5 plan contains invalid FIT/AUDIT roles")
    if set(plan["direct_mpc_variable_count"].astype(int)) != {12 * 109}:
        raise ValueError("V127 D5 plan does not describe the 1308 online MPC variables")
    if (plan.groupby("rainfall_group")["split_role"].nunique() != 1).any():
        raise ValueError("V127 D5 rainfall group crosses FIT/AUDIT")

    checkpoints = pd.read_csv(checkpoint_metadata_path)
    keys = ["event_id", "rainfall_group", "checkpoint_id"]
    required_meta = set(keys + ["inp_path", "trajectory_metadata_path"])
    missing_meta = sorted(required_meta - set(checkpoints.columns))
    if missing_meta:
        raise ValueError(f"V127 D5 checkpoint metadata missing: {missing_meta}")
    meta = checkpoints.copy()
    if "scientific_split" in meta:
        meta = meta[meta["scientific_split"].astype(str).str.lower() == "development"]
    if "development_fold" in meta:
        meta = meta[meta["development_fold"].astype(str).str.lower() == "train"]
    if meta.duplicated(keys).any():
        raise ValueError("V127 D5 checkpoint metadata is not one-to-one")
    frame = plan.merge(
        meta[keys + ["inp_path", "trajectory_metadata_path"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if frame[["inp_path", "trajectory_metadata_path"]].isna().any().any():
        raise ValueError("V127 D5 plan cannot resolve all checkpoint execution assets")

    graph = _load_graph(graph_path)
    actuator_ids = tuple(str(x) for x in graph.actuator_ids)
    if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
        raise ValueError("V127 D5 execution requires exactly 109 frozen actuator IDs")

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        active_target = _vector(
            str(row["active_target_json"]), len(actuator_ids), label="active target"
        )
        center_fraction = _fractions(
            str(row["center_fractions_json"]), len(actuator_ids), label="center fractions"
        )
        probe_fraction = _fractions(
            str(row["probe_fractions_json"]), len(actuator_ids), label="probe fractions"
        )
        settings, blocks = _blocks(
            str(row["action_sequence_json"]), actuator_ids, active_target=active_target
        )
        score_sha = sequence_sha256_v127(np.repeat(blocks, 2, axis=0))
        if score_sha != str(row["action_sequence_sha256"]):
            raise ValueError("V127 D5 plan action-sequence SHA changed before execution")
        elapsed = int(row["elapsed_seconds"])
        if elapsed < 0 or elapsed % 60:
            raise ValueError("V127 D5 elapsed checkpoint is invalid")
        role = str(row["probe_role"])
        direction_fraction_json = str(row["direction_fractions_json"])
        if role == "center":
            if direction_fraction_json not in {"", "nan"}:
                raise ValueError("V127 D5 center unexpectedly carries a direction")
        else:
            direction = _fractions(
                direction_fraction_json,
                len(actuator_ids),
                label="direction fractions",
            )
            # Direction components are signed and therefore are not [0,1]; _fractions is
            # only for bounded variables. Re-parse with the appropriate contract here.
            direction = np.asarray(json.loads(direction_fraction_json), dtype=np.float32)
            if direction.shape != (12, len(actuator_ids)) or not np.isfinite(direction).all():
                raise ValueError("V127 D5 direction must be finite 12 x actuator_count")
            if abs(float(np.linalg.norm(direction)) - 1.0) > 5e-5:
                raise ValueError("V127 D5 direction is not unit-L2 in online variable space")
        records.append(
            {
                "v127_d5_execution_manifest_contract": V127_D5_EXECUTION_CONTRACT,
                "v127_d5_contract": V127_D5_CONTRACT,
                "plan_row_id": str(row["plan_row_id"]),
                "d5_split_role": str(row["split_role"]),
                "event_id": str(row["event_id"]),
                "rainfall_group": str(row["rainfall_group"]),
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_minutes": elapsed // 60,
                "inp_path": str(row["inp_path"]),
                "trajectory_metadata_path": str(row["trajectory_metadata_path"]),
                "data_role": f"D5_V127_{role.upper()}",
                "source_kind": "D5",
                "center_id": str(row["center_id"]),
                "center_family": str(row["center_family"]),
                "probe_role": role,
                "direction_id": str(row["direction_id"]),
                "direction_family": str(row["direction_family"]),
                "direction_fractions_json": direction_fraction_json,
                "center_fractions_json": json.dumps(
                    center_fraction.tolist(), separators=(",", ":")
                ),
                "probe_fractions_json": json.dumps(
                    probe_fraction.tolist(), separators=(",", ":")
                ),
                "active_target_json": json.dumps(
                    active_target.tolist(), separators=(",", ":")
                ),
                "direct_mpc_variable_count": 12 * 109,
                "epsilon": float(row["epsilon"]),
                "sequence_index": 0,
                "settings_sequence_json": json.dumps(
                    settings, sort_keys=True, separators=(",", ":")
                ),
                "sequence_sha256": canonical_sequence_sha(settings),
                "d5_scoring_sequence_sha256": str(row["action_sequence_sha256"]),
                "model_horizon_steps": 72,
                "model_step_seconds": 300,
                "control_update_seconds": 600,
                "control_block_steps": 2,
                "control_blocks": 36,
                "free_control_blocks": int(row["free_control_blocks"]),
                "d3_time_contract": D3_TIME_CONTRACT,
                "d3_feasibility_contract": D3_FEASIBILITY_CONTRACT,
                "sequence_rate_feasible": True,
                "all_actuators_eligible": True,
                "fixed_active_subset": False,
                "future_action_rule": "H120_continuous_free_targets_then_hold_terminal_target_to_H360",
                "rbc_is_action_space_ceiling": False,
            }
        )
    out = pd.DataFrame.from_records(records)
    if out.duplicated(["checkpoint_id", "sequence_sha256"]).any():
        raise RuntimeError("V127 D5 execution manifest contains duplicate checkpoint/action sequences")
    for center_id, group in out.groupby("center_id", sort=False):
        if int((group["probe_role"] == "center").sum()) != 1:
            raise RuntimeError(f"V127 D5 center {center_id} does not have exactly one centre branch")
        pairs = group[group["probe_role"].isin(["plus", "minus"])]
        for direction_id, pair in pairs.groupby("direction_id", sort=False):
            if set(pair["probe_role"].astype(str)) != {"plus", "minus"} or len(pair) != 2:
                raise RuntimeError(f"V127 D5 direction {direction_id} is not one +/- pair")
            eps = pair["epsilon"].astype(float).to_numpy()
            if not np.allclose(eps, eps[0], rtol=0.0, atol=1e-12) or eps[0] <= 0:
                raise RuntimeError(f"V127 D5 direction {direction_id} has inconsistent epsilon")
            plus = np.asarray(
                json.loads(str(pair.loc[pair["probe_role"] == "plus", "probe_fractions_json"].iloc[0])),
                dtype=np.float32,
            )
            minus = np.asarray(
                json.loads(str(pair.loc[pair["probe_role"] == "minus", "probe_fractions_json"].iloc[0])),
                dtype=np.float32,
            )
            center = np.asarray(
                json.loads(str(pair["center_fractions_json"].iloc[0])), dtype=np.float32
            )
            if float(np.max(np.abs(0.5 * (plus + minus) - center))) > 2.0e-6:
                raise RuntimeError(f"V127 D5 direction {direction_id} lost fraction-space symmetry")
    fit = set(out.loc[out["d5_split_role"] == "fit", "rainfall_group"].astype(str))
    audit = set(out.loc[out["d5_split_role"] == "audit", "rainfall_group"].astype(str))
    if fit & audit:
        raise RuntimeError("V127 D5 FIT/AUDIT rainfall leakage")
    summary = {
        "contract": V127_D5_EXECUTION_CONTRACT,
        "rows": len(out),
        "checkpoints": int(out["checkpoint_id"].nunique()),
        "centers": int(out["center_id"].nunique()),
        "gradient_pairs": int((out["probe_role"] == "plus").sum()),
        "gradient_variable_space": "exact online 12x109 L-BFGS-B fraction tensor",
        "direct_mpc_variable_count": 12 * 109,
        "actuators": len(actuator_ids),
        "fit_rainfall_groups": sorted(fit),
        "audit_rainfall_groups": sorted(audit),
        "rainfall_overlap": sorted(fit & audit),
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    args = p.parse_args()
    frame, summary = build_manifest(
        plan_path=args.plan,
        checkpoint_metadata_path=args.checkpoints,
        graph_path=args.graph,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary_path = (
        Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    )
    summary["out"] = str(out.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
