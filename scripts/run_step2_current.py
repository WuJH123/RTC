"""Stable current Project7 Step2 entrypoint.

The primary research target is now explicit: learn how the 109 writable facility actions change
system-wide cumulative TFV, conditioned on the causal Step1 reconstructed state and causal rainfall
forecast.  The current entrypoint therefore trains the direct delta-TFV value model.

The previous V128 hydraulic-world-model curriculum is retained in its versioned scripts for
ablation/history, but it is no longer the canonical path because Development B0 showed poor TFV
candidate ordering despite authoritative SWMM evidence of physical control leverage.
"""
from __future__ import annotations

from run_step2_tfv_value_current import main


if __name__ == "__main__":
    main()
