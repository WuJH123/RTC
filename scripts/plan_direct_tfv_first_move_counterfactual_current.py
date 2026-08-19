"""Select refined target-latch first moves for exact same-prefix authoritative H360 replay.

V11 telemetry stores the exact refined candidate even when admission rejects it.  The candidate is
already the no-further-command target-latch counterfactual: the newly written 109-dimensional target
is repeated until H360, while the reference repeats the previous supervisory target.  This planner
never reconstructs an H10-then-old-HOLD trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_counterfactual import select_counterfactual_decisions
from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS


DIRECT_TFV_FIRST_MOVE_COUNTERFACTUAL_CONTRACT = (
    "PROJECT7_DIRECT_TFV_TARGET_LATCH_FIRST_MOVE_COUNTERFACTUAL_MANIFEST_V1"
)


def _load(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("strategy") != "proposed_direct_tfv_all109_receding_mpc":
        raise ValueError("first-move replay requires Direct-TFV Proposed metadata")
    if metadata.get("refined_first_move_execution_used") is not True:
        raise ValueError("metadata was not produced by target-latch first-move runtime")
    if str(metadata.get("refined_first_move_semantics", "")) != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
        raise ValueError("runtime metadata has the wrong target-latch first-move semantics")
    decision_file = metadata.get("decision_file")
    if not decision_file:
        raise ValueError("metadata lacks decision_file")
    rows = [
        json.loads(line)
        for line in (metadata_path.parent / str(decision_file)).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if int(metadata.get("decisions", -1)) != len(rows):
        raise ValueError("metadata decision count differs from decision log")
    return metadata, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-decisions", type=int, default=8)
    p.add_argument("--latest-elapsed-seconds", type=int)
    args = p.parse_args()

    metadata_path = Path(args.metadata).resolve()
    metadata, rows = _load(metadata_path)
    selected = select_counterfactual_decisions(
        rows,
        max_decisions=int(args.max_decisions),
        latest_elapsed_seconds=args.latest_elapsed_seconds,
    )
    if not selected:
        raise RuntimeError("no refined first-move decisions are available for replay")

    output: list[dict[str, Any]] = []
    for item in selected:
        decision_index = int(item["decision_index"])
        diagnostics = rows[decision_index].get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"decision {decision_index} lacks diagnostics")
        semantics = str(diagnostics.get("counterfactual_candidate_semantics", ""))
        if semantics != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
            raise ValueError(
                f"decision {decision_index} counterfactual semantics are not target-latch V11"
            )
        blocks = [list(map(float, row)) for row in item["optimized_free_control_blocks"]]
        if len(blocks) != 12 or any(len(row) != 109 for row in blocks):
            raise ValueError(f"decision {decision_index} lacks exact [12,109] first-move telemetry")
        first = blocks[0]
        if any(
            max(abs(float(a) - float(b)) for a, b in zip(first, row, strict=True)) > 1.0e-7
            for row in blocks[1:]
        ):
            raise ValueError(
                f"decision {decision_index} target-latch candidate changes after the first command"
            )
        predicted = float(
            diagnostics.get(
                "refined_first_move_predicted_delta_tfv_m3",
                item.get("predicted_delta_tfv_m3", float("nan")),
            )
        )
        if predicted != predicted:
            raise ValueError(f"decision {decision_index} lacks refined first-move prediction")
        payload = dict(item)
        payload.update(
            {
                "predicted_delta_tfv_m3": predicted,
                "refined_first_move_predicted_delta_tfv_m3": predicted,
                "refined_first_move_margin_m3": float(
                    diagnostics.get("refined_first_move_margin_m3", float("nan"))
                ),
                "refined_first_move_upper_bound_m3": float(
                    diagnostics.get("refined_first_move_upper_bound_m3", float("nan"))
                ),
                "refined_first_move_admission_passed": bool(
                    diagnostics.get("refined_first_move_admission_passed", False)
                ),
                "refined_first_move_changed_facility_count": int(
                    diagnostics.get("refined_first_move_changed_facility_count", 0)
                ),
                "counterfactual_candidate_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
            }
        )
        output.append(payload)

    result = {
        "contract": DIRECT_TFV_FIRST_MOVE_COUNTERFACTUAL_CONTRACT,
        "development_only": True,
        "metadata_path": str(metadata_path),
        "source_inp_sha256": metadata.get("source_inp_sha256"),
        "controller_config_sha256": metadata.get("controller_config_sha256"),
        "swmm_engine_version": metadata.get("swmm_engine_version"),
        "step1_model_sha256": metadata.get(
            "step1_model_sha256", metadata.get("step1_model_file_sha256")
        ),
        "step2_model_sha256": metadata.get("step2_model_sha256"),
        "local_reference": "LATCH_PREVIOUS_TARGET_H360",
        "candidate": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
        "scientific_estimand": (
            "authoritative H360 incremental TFV after writing the refined next supervisory target "
            "and leaving that new target latched if no subsequent command arrives, from an identical "
            "causal prefix"
        ),
        "complete_h360_truth_required": True,
        "latest_elapsed_seconds": args.latest_elapsed_seconds,
        "selected_count": len(output),
        "selected": output,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
