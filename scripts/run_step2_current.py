"""Stable current Project7 Step2 entrypoint.

The primary research target is now explicit: learn how the 109 writable facility actions change
system-wide cumulative TFV, conditioned on the causal Step1 reconstructed state and causal rainfall
forecast.  The current entrypoint therefore trains the direct delta-TFV value model.

The previous V128 hydraulic-world-model curriculum is retained in its versioned scripts for
ablation/history, but it is no longer the canonical path because Development B0 showed poor TFV
candidate ordering despite authoritative SWMM evidence of physical control leverage.

Raw Torch Chrome tracing remains fail-closed on the 16-GB Development workstation. The direct
value runner is substantially smaller, but an unbounded trace is still unnecessary for the core
scientific test and previously exhausted host memory.
"""
from __future__ import annotations

import sys

from run_step2_tfv_value_current import main as _direct_main


def main() -> None:
    if (
        "--torch-profiler" in sys.argv[1:]
        and "--help" not in sys.argv[1:]
        and "-h" not in sys.argv[1:]
    ):
        raise RuntimeError(
            "current Project7 disables raw --torch-profiler trace export on the 16-GB "
            "Development workstation. Use normal training outputs and low-frequency GPU/host "
            "telemetry instead."
        )
    _direct_main()


if __name__ == "__main__":
    main()
