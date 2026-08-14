"""Run the V12.2 development controller through the guarded SWMM entrypoint.

V12.2 deliberately reuses the frozen execution-bound V120 finite candidate Value
policy when the stronger continuous-gradient evidence gate is not met.  This wrapper
only swaps the target-latch controller and removes the legacy target-binding proxy;
it never changes candidate scores or projects a selected move after scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtc.controller_v122 import V122_CONTROLLER_CONTRACT, V122TorchMPCController
from rtc.step2_policy_v122 import FirstMoveFinitePolicyV122


class _DirectFinitePolicyV122:
    """Compatibility adapter for the frozen V120 finite candidate policy."""

    accepts_previous_requested_settings = True

    def __init__(self, policy) -> None:
        self.policy = policy
        self.model = policy.model

    def optimize(self, **kwargs):
        return self.policy.optimize(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--step2", required=True)
    known, _ = parser.parse_known_args()
    cfg = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("V122 development wrapper requires the frozen causal V120 config")
    payload = torch.load(known.step2, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("V122 Step2 bundle must be a dictionary")
    candidate_policy = payload.get("candidate_policy")
    if not isinstance(candidate_policy, dict) or candidate_policy.get("continuous_gradient_search") is not False:
        raise ValueError("V122 finite-policy mode requires continuous gradient search disabled")
    gate = payload.get("value_gate")
    if not isinstance(gate, dict) or not bool(gate.get("passed", False)):
        raise ValueError("V122 requires the frozen Step2 Value gate before closed-loop execution")

    import rtc.production_v120_bound as bound

    original_loader = bound.load_value_only_policy_v120

    def load_v122(**kwargs):
        policy = original_loader(**kwargs)
        return FirstMoveFinitePolicyV122(policy, first_move_group_atol=1.0e-7)

    bound.load_value_only_policy_v120 = load_v122
    bound.PreviousTargetBoundPolicyV120 = _DirectFinitePolicyV122
    bound.V120TorchMPCController = V122TorchMPCController

    # The guarded production CLI remains the authority for event-clock, engine,
    # sensor, lineage, and SWMM readback checks.  This wrapper only selects the V122
    # development controller implementation before that guard starts.
    from rtc.production_guard import main as run_policy

    run_policy()


if __name__ == "__main__":
    main()


__all__ = ["V122_CONTROLLER_CONTRACT", "main"]
