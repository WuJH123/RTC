from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .d2_eval import join_manifest_runs
from .units import flow_rate_to_m3s


MAX_FORMAL_PHASE0_SAMPLE_SECONDS = 60
PEAK_CENSOR_FRACTION = 0.90


def _branch_series(
    metadata_path: str | Path,
    actuator_id: str,
    *,
    analysis_horizon_minutes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return time, actuator flow/readback, network flood rate and max depth.

    If ``analysis_horizon_minutes`` is supplied, a longer authoritative trajectory is sliced
    at the exact requested endpoint in memory. No new SWMM branch and no duplicate compact file
    is created. This reuse is valid for trajectory/timing analysis only; cumulative SWMM volume
    truth continues to require an exact endpoint statistics snapshot.
    """

    p = Path(metadata_path)
    meta = json.loads(p.read_text(encoding="utf-8"))
    checkpoint_seconds = int(meta.get("checkpoint_minutes", 0)) * 60
    source_horizon_seconds = int(meta.get("horizon_minutes", 0)) * 60
    if source_horizon_seconds <= 0:
        raise ValueError(f"branch metadata lacks a positive source horizon: {metadata_path}")
    analysis_seconds = (
        source_horizon_seconds
        if analysis_horizon_minutes is None
        else int(analysis_horizon_minutes) * 60
    )
    if analysis_seconds <= 0:
        raise ValueError("analysis horizon must be positive")
    if analysis_seconds > source_horizon_seconds:
        raise ValueError(
            f"requested {analysis_seconds}s timing view exceeds source trajectory "
            f"horizon {source_horizon_seconds}s: {metadata_path}"
        )
    endpoint = checkpoint_seconds + analysis_seconds

    compact = meta.get("compact_file")
    if compact:
        with np.load(p.parent / str(compact), allow_pickle=False) as raw:
            ids = tuple(raw["actuator_ids"].astype(str).tolist())
            if actuator_id not in ids:
                raise ValueError(f"actuator {actuator_id} absent from compact branch {metadata_path}")
            idx = ids.index(actuator_id)
            times = raw["elapsed_seconds"].astype(float)
            keep = times <= endpoint
            if not np.any(keep) or int(round(float(times[keep][-1]))) != endpoint:
                raise ValueError(
                    f"long trajectory lacks exact requested timing-view endpoint {endpoint}s: {metadata_path}"
                )
            state = raw["state_si"][keep].astype(float)
            if state.shape[-1] < 3:
                raise ValueError("compact state lacks depth/flooding channels")
            return (
                times[keep],
                raw["actuator_flow_m3s"][keep, idx].astype(float),
                raw["current_setting"][keep, idx].astype(float),
                np.clip(state[..., 2], 0.0, None).sum(axis=1),
                state[..., 0].max(axis=1),
                source_horizon_seconds,
            )
    name = meta.get("actuator_file")
    node_name = meta.get("node_file")
    if not name or not node_name:
        raise ValueError(f"branch lacks compact data required for Phase0: {metadata_path}")
    act = pd.read_csv(p.parent / str(name), compression="infer")
    node = pd.read_csv(p.parent / str(node_name), compression="infer")
    act = act[
        (act["actuator_id"].astype(str) == str(actuator_id))
        & (act["elapsed_seconds"].astype(int) <= endpoint)
    ].sort_values("elapsed_seconds")
    node = node[node["elapsed_seconds"].astype(int) <= endpoint]
    if act.empty or int(act["elapsed_seconds"].iloc[-1]) != endpoint:
        raise ValueError(f"raw branch lacks exact requested timing-view endpoint: {metadata_path}")
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
    return times, flow, setting, flood, max_depth, source_horizon_seconds


def _first_threshold_time(rel_t: np.ndarray, effect: np.ndarray, threshold: float) -> float:
    hits = np.flatnonzero(effect >= threshold)
    return np.nan if not hits.size else float(rel_t[int(hits[0])])


def _step_response_times(
    t: np.ndarray, effect: np.ndarray
) -> tuple[float, float, float, float, float, bool, float]:
    """Characterise a sustained step-action response without area/horizon confounding."""

    effect = np.asarray(effect, dtype=float)
    rel_t = np.asarray(t, dtype=float) - float(t[0])
    peak = float(effect.max(initial=0.0))
    horizon = float(rel_t[-1]) if rel_t.size else 0.0
    if peak <= 1e-12 or horizon <= 0:
        return peak, np.nan, np.nan, np.nan, np.nan, False, np.nan
    peak_idx = int(np.argmax(effect))
    peak_time = float(rel_t[peak_idx])
    t10 = _first_threshold_time(rel_t, effect, 0.10 * peak)
    t50 = _first_threshold_time(rel_t, effect, 0.50 * peak)
    t90 = _first_threshold_time(rel_t, effect, 0.90 * peak)
    peak_censored = bool(peak_time >= PEAK_CENSOR_FRACTION * horizon)
    endpoint_ratio = float(effect[-1] / peak)
    return peak, t10, t50, t90, peak_time, peak_censored, endpoint_ratio


def _readback_lag_seconds(
    t: np.ndarray, base_setting: np.ndarray, candidate_setting: np.ndarray, requested: float
) -> float:
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
    analysis_horizon_minutes: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if split != "development":
        raise ValueError("Formal Phase0 timing selection is development-only")
    merged = join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" not in merged.columns:
        raise ValueError("Phase0 D2 evidence requires scientific_split lineage")
    merged = merged[merged["scientific_split"].astype(str) == split]
    if merged.empty:
        raise ValueError("no development D2 branches for time-scale analysis")
    keys = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        keys.insert(0, "event_id")
    rows: list[dict[str, object]] = []
    observed_sample_steps: set[int] = set()
    observed_horizons: set[int] = set()
    source_horizons: set[int] = set()

    for _, group in merged.groupby(keys, sort=False):
        base_setting = float(group["base_setting"].iloc[0])
        base_rows = group[np.isclose(group["requested_setting"].astype(float), base_setting)]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        aid = str(base["actuator_id"])
        base_t, base_q, base_u, base_flood, base_depth, source_h = _branch_series(
            str(base["metadata_path"]),
            aid,
            analysis_horizon_minutes=analysis_horizon_minutes,
        )
        step = np.diff(base_t)
        if not step.size or np.any(step <= 0):
            raise ValueError("Phase0 D2 time grid must be strictly increasing")
        if not np.allclose(step, step[0]):
            raise ValueError("Phase0 D2 sampling grid must be regular")
        sample_seconds = int(round(float(step[0])))
        horizon_seconds = int(round(float(base_t[-1] - base_t[0])))
        observed_sample_steps.add(sample_seconds)
        observed_horizons.add(horizon_seconds)
        source_horizons.add(source_h)
        if sample_seconds > max_sample_seconds:
            raise ValueError(
                f"Phase0 sampling is {sample_seconds}s; use <= {max_sample_seconds}s so sub-5-min responses are observable"
            )

        for _, candidate in group.iterrows():
            requested = float(candidate["requested_setting"])
            if np.isclose(requested, base_setting):
                continue
            t, q, u, flood, depth, candidate_source_h = _branch_series(
                str(candidate["metadata_path"]),
                aid,
                analysis_horizon_minutes=analysis_horizon_minutes,
            )
            source_horizons.add(candidate_source_h)
            if not np.array_equal(t, base_t):
                raise ValueError("same-checkpoint D2 branches have different sampling grids")
            q_metrics = _step_response_times(t, np.abs(q - base_q))
            f_metrics = _step_response_times(t, np.abs(flood - base_flood))
            h_metrics = _step_response_times(t, np.abs(depth - base_depth))
            rows.append(
                {
                    "event_id": str(candidate.get("event_id", "")),
                    "checkpoint_id": str(candidate["checkpoint_id"]),
                    "actuator_id": aid,
                    "base_setting": base_setting,
                    "requested_setting": requested,
                    "sample_seconds": sample_seconds,
                    "horizon_seconds": horizon_seconds,
                    "source_simulation_horizon_seconds": candidate_source_h,
                    "readback_separation_lag_seconds": _readback_lag_seconds(t, base_u, u, requested),
                    "peak_abs_flow_effect_m3s": q_metrics[0],
                    "flow_t10_seconds": q_metrics[1],
                    "flow_t50_seconds": q_metrics[2],
                    "flow_t90_seconds": q_metrics[3],
                    "flow_peak_effect_seconds": q_metrics[4],
                    "flow_peak_near_horizon": q_metrics[5],
                    "flow_endpoint_to_peak_ratio": q_metrics[6],
                    "peak_abs_network_flood_rate_effect_m3s": f_metrics[0],
                    "network_flood_t10_seconds": f_metrics[1],
                    "network_flood_t50_seconds": f_metrics[2],
                    "network_flood_t90_seconds": f_metrics[3],
                    "network_flood_peak_effect_seconds": f_metrics[4],
                    "network_flood_peak_near_horizon": f_metrics[5],
                    "network_flood_endpoint_to_peak_ratio": f_metrics[6],
                    "peak_abs_network_max_depth_effect": h_metrics[0],
                    "network_depth_t10_seconds": h_metrics[1],
                    "network_depth_t50_seconds": h_metrics[2],
                    "network_depth_t90_seconds": h_metrics[3],
                    "network_depth_peak_effect_seconds": h_metrics[4],
                    "network_depth_peak_near_horizon": h_metrics[5],
                    "network_depth_endpoint_to_peak_ratio": h_metrics[6],
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

    def censor_fraction(flag_column: str, active_column: str) -> float:
        active = detail[detail[active_column].astype(float) > 1e-12]
        if active.empty:
            return 0.0
        return float(active[flag_column].astype(bool).mean())

    censor = {
        "flow_peak_near_horizon_fraction": censor_fraction(
            "flow_peak_near_horizon", "peak_abs_flow_effect_m3s"
        ),
        "network_flood_peak_near_horizon_fraction": censor_fraction(
            "network_flood_peak_near_horizon", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_depth_peak_near_horizon_fraction": censor_fraction(
            "network_depth_peak_near_horizon", "peak_abs_network_max_depth_effect"
        ),
    }
    horizon_censored = any(value > 0.05 for value in censor.values())
    summary: dict[str, object] = {
        "contract": "PHASE0_D2_STEP_RESPONSE_TIMESCALE_V6_LONG_TRAJECTORY_VIEW",
        "split": split,
        "response_cases": int(len(detail)),
        "actuators_tested": int(detail["actuator_id"].nunique()),
        "sampling_seconds": sorted(observed_sample_steps),
        "horizon_seconds": sorted(observed_horizons),
        "source_simulation_horizon_seconds": sorted(source_horizons),
        "analysis_horizon_minutes": analysis_horizon_minutes,
        "trajectory_prefix_view": bool(analysis_horizon_minutes is not None),
        "cumulative_volume_reuse_authorized": False,
        "formal_max_sampling_seconds": int(max_sample_seconds),
        "readback_separation_lag_seconds": quantiles(
            "readback_separation_lag_seconds", "peak_abs_flow_effect_m3s"
        ),
        "flow_t10_seconds": quantiles("flow_t10_seconds", "peak_abs_flow_effect_m3s"),
        "flow_t50_seconds": quantiles("flow_t50_seconds", "peak_abs_flow_effect_m3s"),
        "flow_t90_seconds": quantiles("flow_t90_seconds", "peak_abs_flow_effect_m3s"),
        "flow_peak_effect_seconds": quantiles(
            "flow_peak_effect_seconds", "peak_abs_flow_effect_m3s"
        ),
        "network_flood_t10_seconds": quantiles(
            "network_flood_t10_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_flood_t90_seconds": quantiles(
            "network_flood_t90_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_flood_peak_effect_seconds": quantiles(
            "network_flood_peak_effect_seconds", "peak_abs_network_flood_rate_effect_m3s"
        ),
        "network_depth_t10_seconds": quantiles(
            "network_depth_t10_seconds", "peak_abs_network_max_depth_effect"
        ),
        "network_depth_t90_seconds": quantiles(
            "network_depth_t90_seconds", "peak_abs_network_max_depth_effect"
        ),
        "network_depth_peak_effect_seconds": quantiles(
            "network_depth_peak_effect_seconds", "peak_abs_network_max_depth_effect"
        ),
        "peak_horizon_censoring": censor,
        "horizon_censored": horizon_censored,
        "automatic_time_scale_selection": False,
        "candidate_production_timing": {
            "model_observation_seconds": 300,
            "control_update_seconds": 600,
            "status": "candidate only; freeze after measured response/readback/runtime review",
        },
        "instruction": (
            "Use <=60s Phase0 sampling. A longer D2 trajectory may be sliced for shorter timing "
            "views instead of rerunning SWMM. Do not reuse its final cumulative node statistics "
            "as shorter-horizon TFV truth unless an exact endpoint snapshot exists. If >5% of "
            "active responses peak in the last 10% of the analysis horizon, the sustained-step "
            "response remains censored. Recovery/decay after releasing an action must be evaluated "
            "with D3/pulse-style sequences rather than by weakening this guard."
        ),
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-frequency D2 step-response audit with reusable long-trajectory views"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--max-sample-seconds", type=int, default=MAX_FORMAL_PHASE0_SAMPLE_SECONDS)
    parser.add_argument(
        "--analysis-horizon-minutes",
        type=int,
        help="slice a longer source D2 trajectory at this exact horizon for timing analysis only",
    )
    args = parser.parse_args()
    detail, summary = build_timescale_report(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        split=args.split,
        max_sample_seconds=args.max_sample_seconds,
        analysis_horizon_minutes=args.analysis_horizon_minutes,
    )
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["horizon_censored"] is True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
