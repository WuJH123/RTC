"""Fail-closed equivalence audit for V125/V126 no-override runs versus Sparse-RBC anchor-only.

If every Proposed decision selected the anchor, the executed target-setting trace must be
the same as an anchor-only run under the same frozen event/runtime lineage. A TFV/PFV
difference in that situation is a Step3/runtime comparability problem, not Step2 model
performance, and must be resolved before claiming local-identification failure.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

CONTRACT = "PROJECT7_V126_ANCHOR_ONLY_EQUIVALENCE_GATE_V1"


def _decisions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty decision trace: {path}")
    return rows


def _priority(path: Path) -> tuple[str, ...]:
    nodes = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(nodes) != 8 or len(set(nodes)) != 8:
        raise ValueError("anchor-equivalence audit requires frozen Priority8")
    return nodes


def _volumes(path: Path, priority: set[str]) -> tuple[float, float]:
    tfv = pfv = 0.0
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"node_id", "delta_flooding_volume_m3"}.issubset(reader.fieldnames):
            raise ValueError(f"invalid node-statistics file: {path}")
        for row in reader:
            node = str(row["node_id"])
            if node in seen:
                raise ValueError(f"duplicate node statistics: {node}")
            seen.add(node)
            value = float(row["delta_flooding_volume_m3"])
            if value != value or value < -1.0e-8:
                raise ValueError(f"invalid flooding volume: {node}={value}")
            value = max(value, 0.0)
            tfv += value
            if node in priority:
                pfv += value
    return float(tfv), float(pfv)


def build_audit(
    *,
    anchor_decisions: Path,
    proposed_decisions: Path,
    anchor_stats: Path,
    proposed_stats: Path,
    priority_nodes: Path,
    settings_atol: float = 1.0e-7,
    volume_atol_m3: float = 1.0e-3,
) -> dict:
    anchor = _decisions(anchor_decisions)
    proposed = _decisions(proposed_decisions)
    if len(anchor) != len(proposed):
        raise ValueError("anchor/proposed decision counts differ")
    max_setting_error = 0.0
    override_rows: list[int] = []
    elapsed_mismatch: list[int] = []
    actuator_mismatch: list[int] = []
    for index, (left, right) in enumerate(zip(anchor, proposed, strict=True)):
        if int(left["elapsed_seconds"]) != int(right["elapsed_seconds"]):
            elapsed_mismatch.append(index)
        left_settings = {str(k): float(v) for k, v in dict(left["settings"]).items()}
        right_settings = {str(k): float(v) for k, v in dict(right["settings"]).items()}
        if set(left_settings) != set(right_settings):
            actuator_mismatch.append(index)
            continue
        max_setting_error = max(
            max_setting_error,
            max(abs(left_settings[key] - right_settings[key]) for key in left_settings),
        )
        diagnostics = dict(right.get("diagnostics") or {})
        if bool(diagnostics.get("learned_override_admitted", False)):
            override_rows.append(index)
    priority = set(_priority(priority_nodes))
    anchor_tfv, anchor_pfv = _volumes(anchor_stats, priority)
    proposed_tfv, proposed_pfv = _volumes(proposed_stats, priority)
    tfv_error = abs(proposed_tfv - anchor_tfv)
    pfv_error = abs(proposed_pfv - anchor_pfv)
    passed = (
        not elapsed_mismatch
        and not actuator_mismatch
        and not override_rows
        and max_setting_error <= float(settings_atol)
        and tfv_error <= float(volume_atol_m3)
        and pfv_error <= float(volume_atol_m3)
    )
    return {
        "contract": CONTRACT,
        "verdict": "ANCHOR_EQUIVALENT" if passed else "ANCHOR_EQUIVALENCE_BLOCKED",
        "passed": passed,
        "decision_count": len(anchor),
        "elapsed_mismatch_rows": elapsed_mismatch,
        "actuator_mismatch_rows": actuator_mismatch,
        "learned_override_rows": override_rows,
        "max_setting_abs_error": max_setting_error,
        "settings_atol": float(settings_atol),
        "anchor_tfv_m3": anchor_tfv,
        "proposed_tfv_m3": proposed_tfv,
        "tfv_abs_error_m3": tfv_error,
        "anchor_pfv_m3": anchor_pfv,
        "proposed_pfv_m3": proposed_pfv,
        "pfv_abs_error_m3": pfv_error,
        "volume_atol_m3": float(volume_atol_m3),
        "interpretation": (
            "When no learned override is admitted, any failed equivalence is a Step3/runtime or run-lineage blocker; "
            "do not attribute the TFV difference to Step2 prediction accuracy."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-decisions", required=True, type=Path)
    p.add_argument("--proposed-decisions", required=True, type=Path)
    p.add_argument("--anchor-node-stats", required=True, type=Path)
    p.add_argument("--proposed-node-stats", required=True, type=Path)
    p.add_argument("--priority-nodes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    payload = build_audit(
        anchor_decisions=args.anchor_decisions,
        proposed_decisions=args.proposed_decisions,
        anchor_stats=args.anchor_node_stats,
        proposed_stats=args.proposed_node_stats,
        priority_nodes=args.priority_nodes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
