from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_design import canonical_action_sha, canonical_sequence_sha


PULSE_DESIGN_CONTRACT = "PHASE0_PULSE_RELEASE_RECOVERY_DESIGN_V2_RATE_VERIFIED"


def _reconstruct_base(row: pd.Series) -> dict[str, float]:
    candidate = {str(k): float(v) for k, v in json.loads(str(row["candidate_settings_json"])).items()}
    actuator_id = str(row["actuator_id"])
    if actuator_id not in candidate:
        raise ValueError(f"D2 candidate lacks its probed actuator {actuator_id}")
    candidate[actuator_id] = float(row["base_setting"])
    expected = str(row["base_action_sha256"])
    actual = canonical_action_sha(candidate)
    if actual != expected:
        raise ValueError(
            "D2 base-action lineage mismatch while designing Phase0 pulse: "
            f"checkpoint={row['checkpoint_id']} expected={expected} reconstructed={actual}"
        )
    return candidate


def _max_action_delta(a: dict[str, float], b: dict[str, float]) -> float:
    if set(a) != set(b):
        raise ValueError("pulse/base complete actuator sets differ")
    return max((abs(float(a[key]) - float(b[key])) for key in a), default=0.0)


def design_pulse_recovery(
    d2_manifest: pd.DataFrame,
    *,
    model_step_seconds: int = 60,
    control_block_seconds: int = 600,
    horizon_minutes: int = 360,
    pulses_per_checkpoint: int = 4,
    max_setting_delta_per_update: float = 0.5,
) -> pd.DataFrame:
    required = {
        "checkpoint_id",
        "checkpoint_minutes",
        "candidate_settings_json",
        "candidate_action_sha256",
        "base_action_sha256",
        "actuator_id",
        "base_setting",
        "requested_setting",
        "trajectory_metadata_path",
        "inp_path",
        "scientific_split",
    }
    missing = sorted(required - set(d2_manifest.columns))
    if missing:
        raise ValueError(f"Phase0 pulse design missing D2 columns: {missing}")
    if (d2_manifest["scientific_split"].astype(str) == "final").any():
        raise ValueError("Phase0 pulse design refuses Final rows")
    if min(model_step_seconds, control_block_seconds, horizon_minutes, pulses_per_checkpoint) <= 0:
        raise ValueError("pulse timing/budget values must be positive")
    if not 0 < max_setting_delta_per_update <= 1.0:
        raise ValueError("max_setting_delta_per_update must be in (0,1]")
    if control_block_seconds % model_step_seconds:
        raise ValueError("pulse control block must be an integer multiple of model step")
    horizon_seconds = horizon_minutes * 60
    if horizon_seconds % control_block_seconds:
        raise ValueError("pulse horizon must contain complete control blocks")
    blocks = horizon_seconds // control_block_seconds
    if blocks < 2:
        raise ValueError("pulse recovery requires at least one pulse and one recovery block")

    records: list[dict[str, object]] = []
    metadata_columns = [
        c
        for c in (
            "event_id",
            "rainfall_group",
            "scientific_split",
            "development_fold",
            "checkpoint_id",
            "checkpoint_minutes",
            "inp_path",
            "trajectory_metadata_path",
        )
        if c in d2_manifest.columns
    ]
    for checkpoint_id, group in d2_manifest.groupby("checkpoint_id", sort=True):
        # Every row must reconstruct to the same complete base action/hash.
        first = group.iloc[0]
        base = _reconstruct_base(first)
        expected_base_sha = str(first["base_action_sha256"])
        for _, source_row in group.iterrows():
            if str(source_row["base_action_sha256"]) != expected_base_sha:
                raise ValueError(f"checkpoint {checkpoint_id} contains multiple base-action hashes")
            if _reconstruct_base(source_row) != base:
                raise ValueError(f"checkpoint {checkpoint_id} reconstructs multiple complete base actions")

        base_sequence = [dict(base) for _ in range(blocks)]
        hold: dict[str, object] = {c: first[c] for c in metadata_columns}
        hold.update(
            {
                "data_role": "PHASE0_PULSE_HOLD_REFERENCE",
                "source_d2_candidate_action_sha256": expected_base_sha,
                "pulse_actuator_id": "",
                "pulse_delta": 0.0,
                "max_step_delta": 0.0,
                "settings_sequence_json": json.dumps(base_sequence, sort_keys=True),
                "sequence_sha256": canonical_sequence_sha(base_sequence),
            }
        )
        records.append(hold)

        noncenter = group[
            ~np.isclose(
                group["requested_setting"].astype(float), group["base_setting"].astype(float)
            )
        ].copy()
        if noncenter.empty:
            continue
        noncenter["abs_delta"] = np.abs(
            noncenter["requested_setting"].astype(float)
            - noncenter["base_setting"].astype(float)
        )
        noncenter = noncenter.sort_values(
            ["abs_delta", "actuator_id", "candidate_action_sha256"],
            ascending=[False, True, True],
        ).drop_duplicates("candidate_action_sha256")
        for _, row in noncenter.head(pulses_per_checkpoint).iterrows():
            pulse = {
                str(k): float(v)
                for k, v in json.loads(str(row["candidate_settings_json"])).items()
            }
            row_base = _reconstruct_base(row)
            if row_base != base:
                raise ValueError(f"checkpoint {checkpoint_id} pulse does not share the frozen base action")
            if canonical_action_sha(pulse) != str(row["candidate_action_sha256"]):
                raise ValueError("D2 candidate action JSON/hash mismatch in Phase0 pulse design")
            change = _max_action_delta(base, pulse)
            release = _max_action_delta(pulse, base)
            max_step_delta = max(change, release)
            if max_step_delta > max_setting_delta_per_update + 1e-12:
                raise ValueError(
                    "Phase0 pulse violates the supervisory setting-rate contract: "
                    f"checkpoint={checkpoint_id}, max_delta={max_step_delta:.6g}, "
                    f"limit={max_setting_delta_per_update:.6g}"
                )
            sequence = [pulse, *[dict(base) for _ in range(blocks - 1)]]
            rec: dict[str, object] = {c: row[c] for c in metadata_columns}
            rec.update(
                {
                    "data_role": "PHASE0_PULSE_RELEASE_RECOVERY",
                    "source_d2_candidate_action_sha256": str(row["candidate_action_sha256"]),
                    "pulse_actuator_id": str(row["actuator_id"]),
                    "pulse_delta": float(row["requested_setting"]) - float(row["base_setting"]),
                    "max_step_delta": float(max_step_delta),
                    "settings_sequence_json": json.dumps(sequence, sort_keys=True),
                    "sequence_sha256": canonical_sequence_sha(sequence),
                }
            )
            records.append(rec)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Phase0 pulse design produced no rows")
    frame["model_step_seconds"] = int(model_step_seconds)
    frame["control_update_seconds"] = int(control_block_seconds)
    frame["control_block_steps"] = int(control_block_seconds // model_step_seconds)
    frame["control_blocks"] = int(blocks)
    frame["model_horizon_steps"] = int(horizon_seconds // model_step_seconds)
    frame["max_setting_delta_per_update"] = float(max_setting_delta_per_update)
    frame["d3_time_contract"] = "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1"
    frame["d3_feasibility_contract"] = "D3_SEQUENTIAL_SETTING_RATE_FEASIBILITY_V1"
    frame["sequence_rate_feasible"] = (
        pd.to_numeric(frame["max_step_delta"], errors="raise")
        <= max_setting_delta_per_update + 1e-12
    )
    if not frame["sequence_rate_feasible"].all():
        raise RuntimeError("Phase0 pulse design produced a rate-infeasible sequence")
    frame["phase0_pulse_contract"] = PULSE_DESIGN_CONTRACT
    return frame


def analyse_pulse_recovery(
    run_summary: pd.DataFrame,
    *,
    control_block_seconds: int,
    recovery_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"checkpoint_id", "data_role", "metadata_path"}
    missing = sorted(required - set(run_summary.columns))
    if missing:
        raise ValueError(f"pulse run summary missing columns: {missing}")
    if not 0 < recovery_fraction < 1:
        raise ValueError("recovery_fraction must be between 0 and 1")
    rows: list[dict[str, object]] = []
    for checkpoint_id, group in run_summary.groupby("checkpoint_id", sort=False):
        hold_rows = group[group["data_role"].astype(str) == "PHASE0_PULSE_HOLD_REFERENCE"]
        if len(hold_rows) != 1:
            raise ValueError(f"checkpoint {checkpoint_id} requires exactly one pulse hold reference")
        hold_meta = Path(str(hold_rows.iloc[0]["metadata_path"]))
        hold = json.loads(hold_meta.read_text(encoding="utf-8"))
        with np.load(hold_meta.parent / str(hold["compact_file"]), allow_pickle=False) as raw:
            base_t = raw["elapsed_seconds"].astype(float)
            base_state = raw["state_si"].astype(float)
            base_flow = raw["actuator_flow_m3s"].astype(float)
        base_flood = np.clip(base_state[..., 2], 0.0, None).sum(axis=1)
        base_depth = base_state[..., 0].max(axis=1)

        for _, item in group.iterrows():
            if str(item["data_role"]) != "PHASE0_PULSE_RELEASE_RECOVERY":
                continue
            meta_path = Path(str(item["metadata_path"]))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(meta_path.parent / str(meta["compact_file"]), allow_pickle=False) as raw:
                t = raw["elapsed_seconds"].astype(float)
                state = raw["state_si"].astype(float)
                flow = raw["actuator_flow_m3s"].astype(float)
            if not np.array_equal(t, base_t):
                raise ValueError("pulse and hold reference time grids differ")
            flood_effect = np.abs(np.clip(state[..., 2], 0.0, None).sum(axis=1) - base_flood)
            depth_effect = np.abs(state[..., 0].max(axis=1) - base_depth)
            flow_effect = np.max(np.abs(flow - base_flow), axis=1)
            release_time = float(t[0] + control_block_seconds)

            def recovery(effect: np.ndarray) -> tuple[float, float | None, float]:
                peak = float(np.max(effect, initial=0.0))
                if peak <= 1e-12:
                    return peak, 0.0, 0.0
                post = np.flatnonzero(
                    (t >= release_time) & (effect <= recovery_fraction * peak)
                )
                seconds = None if not post.size else float(t[int(post[0])] - release_time)
                return peak, seconds, float(effect[-1] / peak)

            f_peak, f_rec, f_end = recovery(flow_effect)
            fl_peak, fl_rec, fl_end = recovery(flood_effect)
            d_peak, d_rec, d_end = recovery(depth_effect)
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "event_id": str(item.get("event_id", "")),
                    "pulse_actuator_id": str(item.get("pulse_actuator_id", "")),
                    "pulse_delta": float(item.get("pulse_delta", 0.0)),
                    "flow_peak_effect": f_peak,
                    "flow_recovery_to_10pct_seconds": f_rec,
                    "flow_endpoint_to_peak_ratio": f_end,
                    "flood_peak_effect": fl_peak,
                    "flood_recovery_to_10pct_seconds": fl_rec,
                    "flood_endpoint_to_peak_ratio": fl_end,
                    "depth_peak_effect": d_peak,
                    "depth_recovery_to_10pct_seconds": d_rec,
                    "depth_endpoint_to_peak_ratio": d_end,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("pulse recovery analysis produced no candidate comparisons")

    def summary(column: str) -> dict[str, float | None]:
        values = pd.to_numeric(detail[column], errors="coerce").dropna().to_numpy(dtype=float)
        return {
            "p50": None if not values.size else float(np.quantile(values, 0.50)),
            "p90": None if not values.size else float(np.quantile(values, 0.90)),
            "unrecovered_fraction": float(detail[column].isna().mean()),
        }

    report = {
        "contract": "PHASE0_PULSE_RELEASE_RECOVERY_ANALYSIS_V1",
        "cases": int(len(detail)),
        "checkpoints": int(detail["checkpoint_id"].nunique()),
        "control_block_seconds": int(control_block_seconds),
        "recovery_threshold_fraction": float(recovery_fraction),
        "flow_recovery": summary("flow_recovery_to_10pct_seconds"),
        "flood_recovery": summary("flood_recovery_to_10pct_seconds"),
        "depth_recovery": summary("depth_recovery_to_10pct_seconds"),
        "interpretation": (
            "This pulse/release diagnostic measures decay after the action is removed. It is "
            "separate from the sustained-D2 peak-near-horizon censor guard and must not be used "
            "to weaken that guard."
        ),
    }
    return detail, report


def design_main() -> None:
    parser = argparse.ArgumentParser(
        description="Design development-only, rate-verified Phase0 pulse/recovery sequences"
    )
    parser.add_argument("--d2-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-step-seconds", type=int, default=60)
    parser.add_argument("--control-block-seconds", type=int, default=600)
    parser.add_argument("--horizon-minutes", type=int, default=360)
    parser.add_argument("--pulses-per-checkpoint", type=int, default=4)
    parser.add_argument("--max-setting-delta-per-update", type=float, default=0.5)
    args = parser.parse_args()
    frame = design_pulse_recovery(
        pd.read_csv(args.d2_manifest),
        model_step_seconds=args.model_step_seconds,
        control_block_seconds=args.control_block_seconds,
        horizon_minutes=args.horizon_minutes,
        pulses_per_checkpoint=args.pulses_per_checkpoint,
        max_setting_delta_per_update=args.max_setting_delta_per_update,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    payload = {
        "contract": PULSE_DESIGN_CONTRACT,
        "rows": int(len(frame)),
        "checkpoints": int(frame["checkpoint_id"].nunique()),
        "pulse_rows": int((frame["data_role"] == "PHASE0_PULSE_RELEASE_RECOVERY").sum()),
        "hold_rows": int((frame["data_role"] == "PHASE0_PULSE_HOLD_REFERENCE").sum()),
        "max_setting_delta_per_update": float(args.max_setting_delta_per_update),
        "max_observed_step_delta": float(pd.to_numeric(frame["max_step_delta"]).max()),
        "all_sequences_rate_feasible": bool(frame["sequence_rate_feasible"].all()),
        "out": str(out),
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def analyse_main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Phase0 action-release recovery timing")
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--control-block-seconds", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    detail, report = analyse_pulse_recovery(
        pd.read_csv(args.run_summary), control_block_seconds=args.control_block_seconds
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out.with_suffix(".detail.csv"), index=False)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
