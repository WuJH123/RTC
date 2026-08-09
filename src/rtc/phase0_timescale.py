from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .d2_eval import join_manifest_runs
from .units import flow_rate_to_m3s


MAX_FORMAL_PHASE0_SAMPLE_SECONDS = 60


def _branch_series(
    metadata_path: str | Path, actuator_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, actuator flow/readback, network flood rate and max depth."""

    p = Path(metadata_path)
    meta = json.loads(p.read_text(encoding="utf-8"))
    compact = meta.get("compact_file")
    if compact:
        with np.load(p.parent / str(compact), allow_pickle=False) as raw:
            ids = tuple(raw["actuator_ids"].astype(str).tolist())
            if actuator_id not in ids:
                raise ValueError(f"actuator {actuator_id} absent from compact branch {metadata_path}")
            idx = ids.index(actuator_id)
            state = raw["state_si"].astype(float)
            if state.shape[-1] < 3:
                raise ValueError("compact state lacks depth/flooding channels")
            return (
                raw["elapsed_seconds"].astype(float),
                raw["actuator_flow_m3s"][:, idx].astype(float),
                raw["current_setting"][:, idx].astype(float),
                np.clip(state[..., 2], 0.0, None).sum(axis=1),
                state[..., 0].max(axis=1),
            )
    # Legacy fallback is retained only for old diagnostics. New Formal Phase0 must use compact.
    name = meta.get("actuator_file")
    node_name = meta.get("node_file")
    if not name or not node_name:
        raise ValueError(f"branch lacks compact data required for Phase0: {metadata_path}")
    act = pd.read_csv(p.parent / str(name), compression="infer")
    node = pd.read_csv(p.parent / str(node_name), compression="infer")
    act = act[act["actuator_id"].astype(str) == str(actuator_id)].sort_values("elapsed_seconds")
    if act.empty:
        raise ValueError(f"actuator {actuator_id} absent from branch {metadata_path}")
    times = act["elapsed_seconds"].to_numpy(dtype=float)
    flow = flow_rate_to_m3s(act["flow"].to_numpy(dtype=float), str(meta["flow_units"]))
    setting = act["current_setting"].to_numpy(dtype=float)
    by_time = node.groupby("elapsed_seconds", sort=True)
    flood = np.asarray([
        float(np.clip(g["flooding"].to_numpy(dtype=float), 0.0, None).sum())
        for _, g in by_time
    ])
    max_depth = np.asarray([float(g["depth"].astype(float).max()) for _, g in by_time])
    flood = flow_rate_to_m3s(flood, str(meta["flow_units"]))
    return times, flow, setting, flood, max_depth


def _effect_timescale(t: np.ndarray, effect: np.ndarray) -> tuple[float, float, float, float]:
    effect = np.asarray(effect, dtype=float)
    peak = float(effect.max(initial=0.0))
    if peak <= 1e-12:
        return peak, np.nan, np.nan, np.nan
    rel_t = t - t[0]
    onset_idx = int(np.argmax(effect >= 0.10 * peak))
    peak_idx = int(np.argmax(effect))
    dt = np.diff(t)
    area = np.cumsum(0.5 * (effect[:-1] + effect[1:]) * dt)
    total = float(area[-1]) if area.size else 0.0
    if total > 0:
        mass_idx = min(
            int(np.searchsorted(area, 0.90 * total, side="left")) + 1,
            len(rel_t) - 1,
        )
        mass90 = float(rel_t[mass_idx])
    else:
        mass90 = np.nan
    return peak, float(rel_t[onset_idx]), float(rel_t[peak_idx]), mass90


def _readback_lag_seconds(
    t: np.ndarray, base_setting: np.ndarray, candidate_setting: np.ndarray, requested: float
) -> float:
    """First time candidate readback materially separates from same-prefix base readback."""

    separation = np.abs(candidate_setting - base_setting)
    requested_delta = abs(float(requested) - float(base_setting[0]))
    threshold = max(0.01, 0.10 * requested_delta)
    hits = np.flatnonzero(separation >= threshold)
    return np.nan if not hits.size else float(t[int(hits[0])] - t[0])


def build_timescale_report(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    split: str = "development",
    max_sample_seconds: int = MAX_FORMAL_PHASE0_SAMPLE_SECONDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == split]
    if merged.empty:
        raise ValueError("no D2 branches for time-scale analysis")
    keys = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        keys.insert(0, "event_id")
    rows: list[dict[str, object]] = []
    observed_sample_steps: set[int] = set()

    for _, group in merged.groupby(keys, sort=False):
        base_setting = float(group["base_setting"].iloc[0])
        base_rows = group[np.isclose(group["requested_setting"].astype(float), base_setting)]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        aid = str(base["actuator_id"])
        base_t, base_q, base_u, base_flood, base_depth = _branch_series(
            str(base["metadata_path"]), aid
        )
        step = np.diff(base_t)
        if not step.size or np.any(step <= 0):
            raise ValueError("Phase0 D2 time grid must be strictly increasing")
        if not np.allclose(step, step[0]):
            raise ValueError("Phase0 D2 sampling grid must be regular")
        sample_seconds = int(round(float(step[0])))
        observed_sample_steps.add(sample_seconds)
        if sample_seconds > max_sample_seconds:
            raise ValueError(
                f"Phase0 sampling is {sample_seconds}s; use <= {max_sample_seconds}s so sub-5-min responses are observable"
            )

        for _, candidate in group.iterrows():
            requested = float(candidate["requested_setting"])
            if np.isclose(requested, base_setting):
                continue
            t, q, u, flood, depth = _branch_series(str(candidate["metadata_path"]), aid)
            if not np.array_equal(t, base_t):
                raise ValueError("same-checkpoint D2 branches have different sampling grids")
            q_peak, q_onset, q_peak_t, q_mass90 = _effect_timescale(t, np.abs(q - base_q))
            f_peak, f_onset, f_peak_t, f_mass90 = _effect_timescale(
                t, np.abs(flood - base_flood)
            )
            h_peak, h_onset, h_peak_t, h_mass90 = _effect_timescale(
                t, np.abs(depth - base_depth)
            )
            rows.append(
                {
                    "event_id": str(candidate.get("event_id", "")),
                    "checkpoint_id": str(candidate["checkpoint_id"]),
                    "actuator_id": aid,
                    "base_setting": base_setting,
                    "requested_setting": requested,
                    "sample_seconds": sample_seconds,
                    "readback_separation_lag_seconds": _readback_lag_seconds(
                        t, base_u, u, requested
                    ),
                    "peak_abs_flow_effect_m3s": q_peak,
                    "flow_response_onset_seconds": q_onset,
                    "flow_peak_effect_seconds": q_peak_t,
                    "flow_response_mass90_seconds": q_mass90,
                    "peak_abs_network_flood_rate_effect_m3s": f_peak,
                    "network_flood_response_onset_seconds": f_onset,
                    "network_flood_peak_effect_seconds": f_peak_t,
                    "network_flood_response_mass90_seconds": f_mass90,
                    "peak_abs_network_max_depth_effect": h_peak,
                    "network_depth_response_onset_seconds": h_onset,
                    "network_depth_peak_effect_seconds": h_peak_t,
                    "network_depth_response_mass90_seconds": h_mass90,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("no non-center D2 responses were available")

    def quantiles(column: str, active_column: str) -> dict[str, float | None]:
        active = detail[detail[active_column].astype(float) > 1e-12]
        values = active[column].dropna().to_numpy(dtype=float)
        if not values.size:
            return {"p10": None, "p50": None, "p90": None}
        return {
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }

    summary: dict[str, object] = {
        "contract": "PHASE0_D2_HYDRAULIC_TIMESCALE_REPORT_V4_HIGH_FREQUENCY",
        "split": split,
        "response_cases": int(len(detail)),
        "actuators_tested": int(detail["actuator_id"].nunique()),
        "sampling_seconds": sorted(observed_sample_steps),
        "formal_max_sampling_seconds": int(max_sample_seconds),
        "readback_separation_lag_seconds": quantiles(
            "readback_separation_lag_seconds", "peak_abs_flow_effect_m3s"
        ),
        "flow_response_onset_seconds": quantiles(
            "flow_response_onset_seconds", "peak_abs_flow_effect_m3s"
        ),
        "flow_peak_effect_seconds": quantiles(
            "flow_peak_effect_seconds", "peak_abs_flow_effect_m3s"
        ),
        "flow_response_mass90_seconds": quantiles(
            "flow_response_mass90_seconds", "peak_abs_flow_effect_m3s"
        ),
        "network_flood_response_onset_seconds": quantiles(
            "network_flood_response_onset_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_flood_peak_effect_seconds": quantiles(
            "network_flood_peak_effect_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_flood_response_mass90_seconds": quantiles(
            "network_flood_response_mass90_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_depth_response_onset_seconds": quantiles(
            "network_depth_response_onset_seconds", "peak_abs_network_max_depth_effect"
        ),
        "automatic_time_scale_selection": False,
        "candidate_production_timing": {
            "model_observation_seconds": 300,
            "control_update_seconds": 600,
            "status": "candidate only; freeze after reviewing this high-frequency report and runtime compute/readback tests",
        },
        "instruction": "Do not infer a 5-min model step from 5-min sampled Phase0 data. Use <=60s Phase0 D2 sampling, then freeze production cadence from measured readback/flow/network response plus online compute latency.",
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-frequency D2 audit of actuator/readback and network hydraulic response time scales"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--max-sample-seconds", type=int, default=MAX_FORMAL_PHASE0_SAMPLE_SECONDS)
    args = parser.parse_args()
    detail, summary = build_timescale_report(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        split=args.split,
        max_sample_seconds=args.max_sample_seconds,
    )
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
