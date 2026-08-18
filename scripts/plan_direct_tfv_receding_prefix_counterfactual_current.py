"""Select V8 decisions for exact same-prefix H360 replay of the executable H10 prefix.

The existing Direct-TFV counterfactual selector samples strong/median/mild raw optimizer queries.
This planner then replaces uncommitted future H120 blocks with the recorded HOLD target so the
protected authoritative replay runner evaluates exactly the V8 execution estimand:
EXECUTE_H10_THEN_HOLD_ACTIVE_TARGET_H350.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_counterfactual import select_counterfactual_decisions
from rtc.direct_tfv_receding_prefix import DIRECT_TFV_RECEDING_PREFIX_SEMANTICS


DIRECT_TFV_RECEDING_PREFIX_COUNTERFACTUAL_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RECEDING_PREFIX_COUNTERFACTUAL_MANIFEST_V1"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _load(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("strategy") != "proposed_direct_tfv_all109_receding_mpc":
        raise ValueError("receding-prefix replay requires current Direct-TFV Proposed metadata")
    if metadata.get("receding_prefix_admission_used") is not True:
        raise ValueError("metadata was not produced by the V8 receding-prefix runtime")
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
    p.add_argument("--max-decisions", type=int, default=6)
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
        raise RuntimeError("no V8 raw optimizer decisions are available for receding-prefix replay")

    output: list[dict[str, Any]] = []
    for item in selected:
        decision_index = int(item["decision_index"])
        diagnostics = rows[decision_index].get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"decision {decision_index} lacks diagnostics")
        prefix_predicted = float(diagnostics.get("receding_prefix_predicted_delta_tfv_m3", float("nan")))
        if not (prefix_predicted == prefix_predicted):
            raise ValueError(f"decision {decision_index} lacks receding-prefix prediction")
        hold = [float(x) for x in item["hold_reference_settings"]]
        full_blocks = item["optimized_free_control_blocks"]
        prefix_blocks = [list(full_blocks[0])] + [list(hold) for _ in range(11)]
        full_plan_hash = str(item["plan_sha256"])
        prefix_hash = _canonical_sha256(
            {
                "counterfactual_actuator_ids": item["counterfactual_actuator_ids"],
                "hold_reference_settings": hold,
                "optimized_free_control_blocks": prefix_blocks,
            }
        )
        payload = dict(item)
        payload.update(
            {
                "full_plan_sha256": full_plan_hash,
                "full_plan_optimized_free_control_blocks": full_blocks,
                "optimized_free_control_blocks": prefix_blocks,
                "plan_sha256": prefix_hash,
                "predicted_delta_tfv_m3": prefix_predicted,
                "receding_prefix_predicted_delta_tfv_m3": prefix_predicted,
                "receding_prefix_admission_passed": bool(
                    diagnostics.get("receding_prefix_admission_passed", False)
                ),
                "receding_prefix_margin_m3": float(
                    diagnostics.get("receding_prefix_margin_m3", float("nan"))
                ),
                "receding_prefix_upper_bound_m3": float(
                    diagnostics.get("receding_prefix_upper_bound_m3", float("nan"))
                ),
                "counterfactual_candidate_semantics": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
            }
        )
        output.append(payload)

    result = {
        "contract": DIRECT_TFV_RECEDING_PREFIX_COUNTERFACTUAL_CONTRACT,
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
        "candidate": DIRECT_TFV_RECEDING_PREFIX_SEMANTICS,
        "scientific_estimand": (
            "authoritative H360 incremental TFV of the exact first H10 target actually committed "
            "before replanning, followed by HOLD H350, from an identical causal prefix"
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
