from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.controller_v121 import V121TorchMPCController
from rtc.step3_policy_v121 import (
    FIRST_MOVE_GROUP_ATOL,
    FirstMoveRobustCandidatePolicyV121,
    V121_STEP3_CONTRACT,
)

V121_CONTROLLER_CONTRACT = "PROJECT7_V121_FIRST_MOVE_ROBUST_CONTROLLER_V1"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    known, _ = parser.parse_known_args()
    cfg = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("v121_contract") != V121_CONTROLLER_CONTRACT:
        raise ValueError("run_policy_v121 requires the frozen V121 controller config")
    step3 = cfg.get("step3")
    if not isinstance(step3, dict) or step3.get("contract") != V121_STEP3_CONTRACT:
        raise ValueError("V121 Step3 contract mismatch")
    if step3.get("tail_aggregation") != "median_within_identical_first_move":
        raise ValueError("V121 tail aggregation drift")
    if abs(float(step3.get("first_move_group_atol", -1.0)) - FIRST_MOVE_GROUP_ATOL) > 1e-15:
        raise ValueError("V121 first-move grouping tolerance drift")

    controller = cfg.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("V121 controller section missing")
    if abs(float(controller.get("readback_target_tolerance", -1.0)) - 1e-6) > 1e-12:
        raise ValueError("V121 target readback tolerance drift")
    if abs(float(controller.get("readback_current_tolerance", -1.0)) - 1.0) > 1e-12:
        raise ValueError("V121 current setting must be treated as physical state, not write latch")

    import rtc.production_v120_bound as bound

    original_loader = bound.load_value_only_policy_v120

    def load_v121(**kwargs):
        base = original_loader(**kwargs)
        return FirstMoveRobustCandidatePolicyV121(
            base,
            first_move_group_atol=FIRST_MOVE_GROUP_ATOL,
        )

    bound.load_value_only_policy_v120 = load_v121
    bound.V120TorchMPCController = V121TorchMPCController

    from rtc.production_guard import main as run_policy

    run_policy()


if __name__ == "__main__":
    main()
