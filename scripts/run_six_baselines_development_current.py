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
from typing import Any

from rtc.baseline_panel import (
    CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT,
    baseline_lineage_failures,
    tfv_m3,
)
from rtc.baselines import FORMAL_FIXED_BASELINE_IDS
from rtc.execution_audit_v127 import audit_target_write_readback_v127


PYTHON_COMMAND_BASELINES = {"auto_rbc", "efd", "all_open", "all_closed"}
SCIENTIFIC_PYTHON_BASELINES = {"auto_rbc", "efd"}


def _decision_execution_summary(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    decision_name = metadata.get("decision_file")
    if not decision_name:
        return {
            "decision_rows": 0,
            "material_projection_decisions": 0,
            "numerical_equivalence_decisions": 0,
            "maximum_command_delta_from_previous_target": 0.0,
            "rule_contract_values": [],
            "efd_depth_fallback_storage_count_max": 0,
        }
    decision_path = metadata_path.parent / str(decision_name)
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    rows = [
        json.loads(line)
        for line in decision_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    material = numerical = 0
    maximum_delta = 0.0
    rule_contracts: set[str] = set()
    efd_depth_fallback_max = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("baseline decision log contains a non-object row")
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        material += int(diagnostics.get("continuity_projection_applied") is True)
        numerical += int(diagnostics.get("continuity_numerical_equivalence_applied") is True)
        delta = diagnostics.get("command_delta_from_previous_target_max")
        if delta is not None:
            maximum_delta = max(maximum_delta, float(delta))
        rule_contract = str(diagnostics.get("rule_contract", "")).strip()
        if rule_contract:
            rule_contracts.add(rule_contract)
        efd_depth_fallback_max = max(
            efd_depth_fallback_max,
            int(diagnostics.get("depth_fallback_storage_count", 0)),
        )
    return {
        "decision_rows": len(rows),
        "material_projection_decisions": material,
        "numerical_equivalence_decisions": numerical,
        "maximum_command_delta_from_previous_target": maximum_delta,
        "rule_contract_values": sorted(rule_contracts),
        "efd_depth_fallback_storage_count_max": efd_depth_fallback_max,
    }


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
    execution = _decision_execution_summary(metadata_path, metadata)
    if (
        strategy in SCIENTIFIC_PYTHON_BASELINES
        and int(execution["material_projection_decisions"]) != 0
    ):
        raise RuntimeError(
            f"{strategy}: scientific comparator required material continuity projection; "
            "its native rule command is not satisfying the declared 0.5 target-slew contract"
        )
    if strategy == "efd" and int(execution["efd_depth_fallback_storage_count_max"]) != 0:
        raise RuntimeError(
            "efd: storage-volume EFD degraded to normalized-depth fallback; repair/freeze valid "
            "storage geometry before using this run as a scientific comparator"
        )
    sensor_nodes = metadata.get("sensor_nodes")
    sensor_count = len(sensor_nodes) if isinstance(sensor_nodes, list) else 0
    return {
        "event_id": event_id,
        "strategy": strategy,
        "tfv_m3": tfv_m3(statistics_path),
        "global_peak_flood_rate_m3s": float(metadata.get("global_peak_flood_rate_m3s", 0.0)),
        "flow_routing_error_pct": float(metadata.get("flow_routing_error_pct", 0.0)),
        "decisions": int(metadata.get("decisions", 0)),
        "sensor_node_count": sensor_count,
        "target_write_readback_passed": (
            None if write_audit is None else bool(write_audit.get("passed"))
        ),
        **execution,
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
        "scientific_comparator_engineering_projection_required": False,
        "efd_storage_volume_semantics_verified": True,
        "baseline_information_basis_reported": True,
        "global_peak_role": "report_only",
        "rows": rows,
    }
    out = root / "SIX_BASELINE_DEVELOPMENT_PANEL_CURRENT.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
