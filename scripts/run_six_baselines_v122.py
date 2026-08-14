"""Run the six fixed Project7 baseline policies on one prepared event.

This script is intentionally independent from Proposed.  Baselines are external
comparators; they are never used as the Proposed online action space.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

BASELINES = (
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
)


def _tfv_m3(statistics_path: Path) -> float:
    total = 0.0
    with gzip.open(statistics_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += float(row["delta_flooding_volume_m3"])
    return float(total)


def _run_one(
    *,
    strategy: str,
    inp: Path,
    sensors: Path,
    config: Path,
    native_controls_template: Path,
    root: Path,
    runtime_cache: Path,
    event_id: str,
) -> dict[str, object]:
    out = root / strategy
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{event_id}__{strategy}"
    command = [
        sys.executable,
        "-m",
        "rtc.production_guard",
        "--strategy",
        strategy,
        "--inp",
        str(inp),
        "--out-dir",
        str(out),
        "--run-id",
        run_id,
        "--sensors",
        str(sensors),
        "--config",
        str(config),
        "--runtime-inp-cache-dir",
        str(runtime_cache),
    ]
    if strategy == "internal_rtc":
        command.extend(["--native-controls-template", str(native_controls_template)])
    subprocess.run(command, check=True)

    metadata_path = out / f"{run_id}.json"
    statistics_path = out / f"{run_id}.node_statistics.csv.gz"
    if not metadata_path.is_file() or not statistics_path.is_file():
        raise RuntimeError(f"{strategy}: authoritative SWMM outputs are incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "event_id": event_id,
        "strategy": strategy,
        "tfv_m3": _tfv_m3(statistics_path),
        "global_peak_flood_rate_m3s": float(metadata["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(metadata["flow_routing_error_pct"]),
        "decisions": int(metadata["decisions"]),
        "metadata_path": str(metadata_path.resolve()),
        "node_statistics_path": str(statistics_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run No-control/Internal/Auto-RBC/EFD/All-open/All-closed with identical SWMM event timing"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--native-controls-template", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    inp = Path(args.inp).resolve()
    sensors = Path(args.sensors).resolve()
    config = Path(args.config).resolve()
    native = Path(args.native_controls_template).resolve()
    for required in (inp, sensors, config, native):
        if not required.is_file():
            raise FileNotFoundError(required)

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_cache = root / "_runtime_inp"
    rows = [
        _run_one(
            strategy=strategy,
            inp=inp,
            sensors=sensors,
            config=config,
            native_controls_template=native,
            root=root,
            runtime_cache=runtime_cache,
            event_id=str(args.event_id),
        )
        for strategy in BASELINES
    ]
    no_control = next(float(row["tfv_m3"]) for row in rows if row["strategy"] == "no_control")
    for row in rows:
        tfv = float(row["tfv_m3"])
        row["delta_tfv_vs_no_control_m3"] = tfv - no_control
        row["tfv_reduction_vs_no_control_pct"] = (
            100.0 * (no_control - tfv) / no_control if no_control > 0.0 else 0.0
        )

    csv_path = root / "BASELINE_COMPARISON_V122.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "contract": "PROJECT7_V122_FIXED_SIX_BASELINE_SWMM_COMPARISON_V1",
        "event_id": str(args.event_id),
        "prepared_inp": str(inp),
        "strategies": list(BASELINES),
        "rows": rows,
        "comparison_csv": str(csv_path.resolve()),
    }
    json_path = root / "BASELINE_COMPARISON_V122.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
