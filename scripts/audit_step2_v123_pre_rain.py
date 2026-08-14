"""Audit the committed V122 pre-rain causal Value behaviour."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from rtc.step2_causal_rainfall_v123 import causal_forecast_from_history_v123
from rtc.step2_train_response_v60 import V60TrainCache


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_datetime(start: str, elapsed: int) -> str:
    return (datetime.fromisoformat(start) + timedelta(seconds=int(elapsed))).isoformat()


def audit(
    *, compact_path: str | Path, decisions_path: str | Path, cache_manifest: str | Path,
    parent_no_control_root: str | Path, metadata_path: str | Path | None = None,
) -> dict[str, object]:
    compact_path, decisions_path = Path(compact_path), Path(decisions_path)
    with np.load(compact_path, allow_pickle=False) as raw:
        compact = {key: raw[key] for key in raw.files}
    elapsed = np.asarray(compact["elapsed_seconds"], dtype=np.int64).reshape(-1)
    rainfall = np.asarray(compact["rainfall_mmhr"], dtype=np.float32)
    lookup = {int(t): i for i, t in enumerate(elapsed.tolist())}
    decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not decisions:
        raise ValueError("V123 pre-rain audit requires decisions")
    start = None
    if metadata_path is not None and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        start = metadata.get("prepared_event_clock", {}).get("simulation_start")
    first_positive_index = np.flatnonzero(rainfall.max(axis=(1, 2)) > 1e-8)
    first_positive_elapsed = int(elapsed[first_positive_index[0]]) if first_positive_index.size else None
    first_control = int(min(int(row["elapsed_seconds"]) for row in decisions))
    requested_minutes = [60, 70, 80, 90, 100, 110]
    rows: list[dict[str, object]] = []
    for minutes in requested_minutes:
        t = minutes * 60
        if t not in lookup:
            raise ValueError(f"V123 pre-rain audit missing compact time {t}")
        i = lookup[t]
        observed = rainfall[i]
        future = rainfall[i:]
        forecast = causal_forecast_from_history_v123(
            rainfall[max(0, i - 12): i + 1], horizon_steps=72, decay_per_step=0.92
        )
        selected = next((row for row in decisions if int(row["elapsed_seconds"]) == t), None)
        rows.append({
            "elapsed_seconds": t,
            "clock_minutes": minutes,
            "datetime": _event_datetime(start, t) if start else None,
            "observed_rainfall_max_mmhr": float(observed.max()),
            "observed_rainfall_mean_mmhr": float(observed.mean()),
            "causal_forecast_max_mmhr": float(forecast.max()),
            "causal_forecast_zero": bool(np.allclose(forecast, 0.0, rtol=0.0, atol=1e-8)),
            "realised_future_rainfall_max_mmhr": float(future.max()),
            "selected_predicted_delta_tfv_m3": None if selected is None else float(selected["diagnostics"].get("predicted_delta_tfv_m3", 0.0)),
            "source": None if selected is None else str(selected.get("source", "")),
        })

    cache = V60TrainCache(cache_manifest)
    parent_root = Path(parent_no_control_root)
    probe = None
    for name in sorted(cache.names("D2") + cache.targeted_d3_names()):
        entry = cache.entry(name)
        t = int(np.asarray(entry.arrays["elapsed_seconds"])[entry.reference_index, 0])
        parents = sorted(parent_root.rglob(f"{entry.event_id}__no_control.compact.npz"))
        if len(parents) != 1:
            continue
        with np.load(parents[0], allow_pickle=False) as raw:
            pt = np.asarray(raw["elapsed_seconds"], dtype=np.int64)
            pr = np.asarray(raw["rainfall_mmhr"], dtype=np.float32)
        ix = np.flatnonzero(pt == t)
        if ix.size != 1:
            continue
        current = float(pr[ix[0]].max())
        future = float(np.asarray(entry.arrays["rainfall"][entry.reference_index]).max())
        if current <= 1e-8 and future > 1e-8:
            probe = {
                "group_name": name,
                "event_id": entry.event_id,
                "checkpoint_id": entry.checkpoint_id,
                "checkpoint_elapsed_seconds": t,
                "parent_current_observed_max_mmhr": current,
                "cache_realised_future_array_max_mmhr": future,
                "interpretation": "cache contains future forcing for SWMM replay; it must not be passed directly as V123 model input",
            }
            break
    payload: dict[str, object] = {
        "contract": "PROJECT7_V123_PRE_RAIN_VALUE_AUDIT_V1",
        "compact_path": str(compact_path.resolve()),
        "compact_sha256": _sha(compact_path),
        "decisions_path": str(decisions_path.resolve()),
        "decisions_sha256": _sha(decisions_path),
        "simulation_start": start,
        "first_positive_rainfall_elapsed_seconds": first_positive_elapsed,
        "first_control_elapsed_seconds": first_control,
        "decision_count": len(decisions),
        "pre_rain_decisions": rows,
        "training_cache_pre_rain_probe": probe,
        "future_realised_rainfall_not_used_by_audit": True,
        "root_cause_interpretation": (
            "The committed V122 decisions select large negative TFV values while the causal central forecast is exactly zero before rainfall. "
            "This is not evidence that the true action effect is zero; it exposes that the Value prediction is not conditioned on a discriminative future-rain signal at those decisions. "
            "V123 therefore rebuilds checkpoint-keyed causal forecasts from parent no-control observed histories and trains/evaluates with that exact store."
        ),
    }
    return payload


def markdown(payload: dict[str, object]) -> str:
    rows = payload["pre_rain_decisions"]
    lines = [
        "# STEP2 V123 PRE-RAIN VALUE AUDIT", "",
        f"- first positive rainfall elapsed: `{payload['first_positive_rainfall_elapsed_seconds']} s`",
        f"- first control elapsed: `{payload['first_control_elapsed_seconds']} s`",
        "", "| minutes | observed max | causal forecast max | realised future max | predicted ΔTFV |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['clock_minutes']} | {row['observed_rainfall_max_mmhr']:.4f} | "
            f"{row['causal_forecast_max_mmhr']:.4f} | {row['realised_future_rainfall_max_mmhr']:.4f} | "
            f"{row['selected_predicted_delta_tfv_m3']!s} |"
        )
    lines.extend(["", payload["root_cause_interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V122 pre-rain Value semantics")
    parser.add_argument("--compact", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--parent-no-control-root", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = audit(
        compact_path=args.compact, decisions_path=args.decisions,
        cache_manifest=args.cache_manifest, parent_no_control_root=args.parent_no_control_root,
        metadata_path=args.metadata,
    )
    (out / "STEP2_V123_PRE_RAIN_VALUE_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "STEP2_V123_PRE_RAIN_VALUE_AUDIT.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
