"""Select at most six Direct-TFV decisions for exact authoritative SWMM branch replay.

The input must be a completed authoritative runtime produced after counterfactual plan telemetry was
added. The output is a diagnostic manifest only: it does not run SWMM and never changes policy
selection. Each selected row contains the exact H360 HOLD reference and the exact 12x109 H120 free
control blocks scored by Step3; terminal hold through H360 is part of the recorded semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_counterfactual import (
    DIRECT_TFV_COUNTERFACTUAL_MANIFEST_CONTRACT,
    select_counterfactual_decisions,
)


def _load(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("strategy") != "proposed_direct_tfv_all109_receding_mpc":
        raise ValueError("counterfactual manifest requires current Direct-TFV Proposed metadata")
    decision_file = metadata.get("decision_file")
    if not decision_file:
        raise ValueError("metadata lacks decision_file")
    decision_path = metadata_path.parent / str(decision_file)
    rows = [
        json.loads(line)
        for line in decision_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("decision log contains non-object row")
    if int(metadata.get("decisions", -1)) != len(rows):
        raise ValueError("metadata decision count differs from decision log")
    return metadata, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-decisions", type=int, default=6)
    parser.add_argument(
        "--latest-elapsed-seconds",
        type=int,
        help=(
            "latest decision elapsed time with a complete H360 truth window; set to "
            "simulation_end_elapsed-21600"
        ),
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    metadata, rows = _load(metadata_path)
    selected = select_counterfactual_decisions(
        rows,
        max_decisions=int(args.max_decisions),
        latest_elapsed_seconds=args.latest_elapsed_seconds,
    )
    if not selected:
        raise RuntimeError(
            "no action decision contains current counterfactual-plan telemetry inside the requested "
            "complete-H360 window; rerun the Development event with the current runtime or revise "
            "the truth-window bound from the authoritative INP clock"
        )
    payload = {
        "contract": DIRECT_TFV_COUNTERFACTUAL_MANIFEST_CONTRACT,
        "development_only": True,
        "metadata_path": str(metadata_path),
        "source_inp_sha256": metadata.get("source_inp_sha256"),
        "controller_config_sha256": metadata.get("controller_config_sha256"),
        "swmm_engine_version": metadata.get("swmm_engine_version"),
        "step1_model_sha256": metadata.get(
            "step1_model_sha256", metadata.get("step1_model_file_sha256")
        ),
        "step2_model_sha256": metadata.get("step2_model_sha256"),
        "local_reference": "HOLD_ACTIVE_TARGET_H360",
        "candidate": "EXACT_OPTIMIZED_H120_FREE_BLOCKS_THEN_TERMINAL_HOLD_H360",
        "scientific_estimand": (
            "authoritative H360 incremental TFV(candidate plan - HOLD) from the identical causal prefix"
        ),
        "whole_event_no_control_reduction_is_not_this_estimand": True,
        "complete_h360_truth_required": True,
        "latest_elapsed_seconds": args.latest_elapsed_seconds,
        "available_action_decisions": sum(
            str(row.get("source", "")) == "MPC_DIRECT_TFV_RECEDING" for row in rows
        ),
        "selected_count": len(selected),
        "selected": selected,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
