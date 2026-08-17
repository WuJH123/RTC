"""Run a fresh provenance-complete six-baseline Development panel for one prepared event.

This is intentionally separate from the blocked seven-strategy production entrypoint. It runs only
fixed external comparators and never promotes the Development Direct-TFV checkpoint. Historical
baseline cache entries are not mutated or retroactively attested; use a fresh output directory when
legacy metadata lacks the current lineage fields.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rtc.baseline_panel import baseline_lineage_failures, tfv_m3
from rtc.baselines import FORMAL_FIXED_BASELINE_IDS
from rtc.execution_audit_v127 import audit_target_write_readback_v127


CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT = (
    "PROJECT7_CURRENT_SIX_FIXED_BASELINE_DEVELOPMENT_SWMM_V1"
)
PYTHON_COMMAND_BASELINES = {"auto_rbc", "efd", "all_open", "all_closed"}


def _run_one(
    *,
    strategy: str,
    inp: Path,
    sensors: Path,
    config: Path,
    native_controls_template: Path,
    root: Path,
    event_id: str,
) -> dict[str, object]:
    out = root / strategy
    run_id = f"{event_id}__{strategy}"
    metadata_path = out / f"{run_id}.json"
    statistics_path = out / f"{run_id}.node_statistics.csv.gz"
    if metadata_path.exists() or statistics_path.exists():
        raise FileExistsError(
            f"refusing to overwrite baseline evidence for {strategy}; choose a fresh --out-dir"
        )
    out.mkdir(parents=True, exist_ok=True)
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
        str(root / "_runtime_inp"),
    ]
    if strategy == "internal_rtc":
        command.extend(["--native-controls-template", str(native_controls_template)])
    subprocess.run(command, check=True)
    if not metadata_path.is_file() or not statistics_path.is_file():
        raise RuntimeError(f"{strategy}: authoritative SWMM outputs are incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{strategy}: metadata must be a JSON object")
    if str(metadata.get("strategy", "")) != strategy:
        raise ValueError(f"{strategy}: production guard did not stamp canonical strategy")
    for field in (
        "source_inp_sha256",
        "controller_config_sha256",
        "swmm_engine_version",
        "prepared_event_clock",
    ):
        if metadata.get(field) in (None, ""):
            raise ValueError(f"{strategy}: fresh baseline metadata lacks {field}")
    write_audit = None
    if strategy in PYTHON_COMMAND_BASELINES:
        write_audit = audit_target_write_readback_v127(metadata_path=metadata_path)
        if write_audit.get("passed") is not True:
            raise RuntimeError(f"{strategy}: target write/readback audit failed")
    return {
        "event_id": event_id,
        "strategy": strategy,
        "tfv_m3": tfv_m3(statistics_path),
        "global_peak_flood_rate_m3s": float(metadata.get("global_peak_flood_rate_m3s", 0.0)),
        "flow_routing_error_pct": float(metadata.get("flow_routing_error_pct", 0.0)),
        "decisions": int(metadata.get("decisions", 0)),
        "target_write_readback_passed": (
            None if write_audit is None else bool(write_audit.get("passed"))
        ),
        "metadata_path": str(metadata_path.resolve()),
        "node_statistics_path": str(statistics_path.resolve()),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    rows = [
        _run_one(
            strategy=strategy,
            inp=inp,
            sensors=sensors,
            config=config,
            native_controls_template=native,
            root=root,
            event_id=str(args.event_id),
        )
        for strategy in FORMAL_FIXED_BASELINE_IDS
    ]

    reference = rows[0]["metadata"]
    assert isinstance(reference, dict)
    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        failures = baseline_lineage_failures(
            proposed=reference,
            baseline=metadata,
            expected_strategy=str(row["strategy"]),
        )
        # The reference itself has strategy=no_control, while the helper deliberately checks only
        # the baseline strategy. All physical/event lineage fields must nevertheless match.
        failures = [failure for failure in failures if not failure.startswith("strategy=")]
        if failures:
            raise RuntimeError(
                f"{row['strategy']}: six-baseline panel lineage differs: " + "; ".join(failures)
            )
        row.pop("metadata")

    no_control_tfv = next(float(row["tfv_m3"]) for row in rows if row["strategy"] == "no_control")
    for row in rows:
        tfv = float(row["tfv_m3"])
        row["delta_tfv_vs_no_control_m3"] = tfv - no_control_tfv
        row["tfv_reduction_vs_no_control_pct"] = (
            100.0 * (no_control_tfv - tfv) / no_control_tfv if no_control_tfv > 0.0 else None
        )

    payload = {
        "contract": CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT,
        "development_only": True,
        "event_id": str(args.event_id),
        "prepared_inp": str(inp),
        "native_controls_template": str(native),
        "strategies": list(FORMAL_FIXED_BASELINE_IDS),
        "baseline_provenance_verified_all": True,
        "global_peak_role": "report_only",
        "rows": rows,
    }
    out = root / "SIX_BASELINE_DEVELOPMENT_PANEL_CURRENT.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
