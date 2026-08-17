"""Stable current Project7 Step2 entrypoint.

The current scientific target is direct and control-oriented: learn how all 109 writable facility
actions change system-wide cumulative TFV from causal reconstructed state and causal rainfall.
MAIN learns facility effects with facility-balanced regression, JOINT learns interactions, and the
final CONTROL stage is restricted to D3 HOLD-reference data because that is the online Step3
reference. D4 is diagnostic only and cannot block Step3.
"""
from __future__ import annotations

import sys

from run_step2_tfv_value_core_current import main as _direct_main


def main() -> None:
    if (
        "--torch-profiler" in sys.argv[1:]
        and "--help" not in sys.argv[1:]
        and "-h" not in sys.argv[1:]
    ):
        raise RuntimeError(
            "current Project7 disables raw --torch-profiler trace export on the 16-GB Development "
            "workstation. Use bounded TRAINING_TELEMETRY.jsonl-style resource telemetry instead."
        )
    _direct_main()


if __name__ == "__main__":
    main()
