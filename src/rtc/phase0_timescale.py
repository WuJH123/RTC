from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .units import flow_rate_to_m3s
from .validation_cli import _join_manifest_runs


def _actuator_flow(metadata_path: str | Path, actuator_id: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(metadata_path)
    meta = json.loads(p.read_text(encoding="utf-8"))
    compact = meta.get("compact_file")
    if compact:
        with np.load(p.parent / str(compact), allow_pickle=False) as raw:
            ids = tuple(raw["actuator_ids"].astype(str).tolist())
            if actuator_id not in ids:
                raise ValueError(f"actuator {actuator_id} absent from compact branch {metadata_path}")
            idx = ids.index(actuator_id)
            return (
                raw["elapsed_seconds"].astype(float),
                raw["actuator_flow_m3s"][:, idx].astype(float),
            )
    # Legacy fallback only; new Formal data must use compact V2.
    name = meta.get("actuator_file")
    if not name:
        raise ValueError(f"branch has neither compact_file nor actuator_file: {metadata_path}")
    frame = pd.read_csv(p.parent / str(name), compression="infer")
    frame = frame[frame["actuator_id"].astype(str) == str(actuator_id)].sort_values("elapsed_seconds")
    if frame.empty:
        raise ValueError(f"actuator {actuator_id} absent from branch {metadata_path}")
    return (
        frame["elapsed_seconds"].to_numpy(dtype=float),
        flow_rate_to_m3s(frame["flow"].to_numpy(dtype=float), str(meta["flow_units"])),
    )


def build_timescale_report(
    *, manifest_path: str | Path, run_summary_path: str | Path, split: str = "development"
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = _join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == split]
    if merged.empty:
        raise ValueError("no D2 branches for time-scale analysis")
    keys = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        keys.insert(0, "event_id")
    rows: list[dict[str, object]] = []
    for _, group in merged.groupby(keys, sort=False):
        base_setting = float(group["base_setting"].iloc[0])
        base_rows = group[np.isclose(group["requested_setting"].astype(float), base_setting)]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        aid = str(base["actuator_id"])
        base_t, base_q = _actuator_flow(str(base["metadata_path"]), aid)
        for _, candidate in group.iterrows():
            if np.isclose(float(candidate["requested_setting"]), base_setting):
                continue
            t, q = _actuator_flow(str(candidate["metadata_path"]), aid)
            if not np.array_equal(t, base_t):
                raise ValueError("same-checkpoint D2 branches have different sampling grids")
            effect = np.abs(q - base_q)
            peak = float(effect.max(initial=0.0))
            if peak <= 1e-12:
                onset_s = peak_s = mass90_s = np.nan
            else:
                rel_t = t - t[0]
                onset_idx = int(np.argmax(effect >= 0.10 * peak))
                peak_idx = int(np.argmax(effect))
                dt = np.diff(t)
                area = np.cumsum(0.5 * (effect[:-1] + effect[1:]) * dt)
                total = float(area[-1]) if area.size else 0.0
                if total > 0:
                    mass_idx = min(int(np.searchsorted(area, 0.90 * total, side="left")) + 1, len(rel_t)-1)
                    mass90_s = float(rel_t[mass_idx])
                else:
                    mass90_s = np.nan
                onset_s = float(rel_t[onset_idx])
                peak_s = float(rel_t[peak_idx])
            rows.append({
                "event_id": str(candidate.get("event_id", "")),
                "checkpoint_id": str(candidate["checkpoint_id"]),
                "actuator_id": aid,
                "base_setting": base_setting,
                "requested_setting": float(candidate["requested_setting"]),
                "peak_abs_flow_effect_m3s": peak,
                "response_onset_seconds": onset_s,
                "peak_effect_seconds": peak_s,
                "response_mass90_seconds": mass90_s,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("no non-center D2 responses were available")
    active = detail[detail["peak_abs_flow_effect_m3s"] > 1e-12]

    def quantiles(column: str) -> dict[str, float | None]:
        values = active[column].dropna().to_numpy(dtype=float)
        return ({"p10": None, "p50": None, "p90": None} if not values.size else {
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        })

    summary: dict[str, object] = {
        "contract": "PHASE0_D2_HYDRAULIC_TIMESCALE_REPORT_V3_COMPACT",
        "split": split,
        "response_cases": int(len(detail)),
        "hydraulically_active_cases": int(len(active)),
        "actuators_with_active_response": int(active["actuator_id"].nunique()),
        "response_onset_seconds": quantiles("response_onset_seconds"),
        "peak_effect_seconds": quantiles("peak_effect_seconds"),
        "response_mass90_seconds": quantiles("response_mass90_seconds"),
        "automatic_time_scale_selection": False,
        "instruction": "Freeze model/control/horizon only after development review; do not inherit Project6 5/10/120 by default.",
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Report empirical compact-D2 hydraulic response time scales")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--split", default="development")
    args = parser.parse_args()
    detail, summary = build_timescale_report(
        manifest_path=args.manifest, run_summary_path=args.run_summary, split=args.split
    )
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
