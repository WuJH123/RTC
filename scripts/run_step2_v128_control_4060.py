"""Run the V127 streaming Step2 pipeline with the V128 control-identifiability profile.

Why this wrapper exists
-----------------------
V127 already contains the memory-safe 16-GB-RAM / 8-GB-VRAM implementation.  Recopying
that large runner would create another stale scientific pipeline.  V128 therefore loads
that canonical runner, changes only two explicitly versioned execution/training choices,
and records both in the normal report:

1. Keep action-order supervision down to the existing 1 m3 absolute SWMM effect floor,
   instead of additionally discarding pairs below 0.1% of reference-event TFV.
2. Default FP32 matrix multiplication to PyTorch ``high`` precision so RTX 4060 tensor
   cores can be used.  Set RTC_V128_MATMUL_PRECISION=highest for strict FP32 comparison.

No AMP is enabled.  No SWMM labels, rainfall splits, causal inputs, horizons or objectives
are changed by this wrapper.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from rtc.v128_control_profile import (
    V128_CONTROL_PROFILE_CONTRACT,
    build_v128_control_training_design,
    configure_v128_cuda_matmul_precision,
)

V128_STREAMING_RUN_CONTRACT = (
    "PROJECT7_V128_CONTROL_IDENTIFIABILITY_4060_STREAMING_V1"
)


def _load_v127_runner() -> ModuleType:
    path = Path(__file__).with_name("run_step2_v127_control_streaming.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_streaming_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 streaming runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    execution_profile = configure_v128_cuda_matmul_precision()
    runner = _load_v127_runner()

    # Keep one canonical implementation of the memory-safe training/evaluation loop.
    # Only the V128 design factory is substituted; all other scientific semantics remain
    # in the V127 runner and therefore continue to receive its existing regression tests.
    runner.V127ControlTrainingDesign = build_v128_control_training_design
    runner.V127_STREAMING_RUN_CONTRACT = V128_STREAMING_RUN_CONTRACT

    base_hardware = runner._hardware

    def hardware_with_v128_profile(device):
        payload = dict(base_hardware(device))
        payload.update(
            {
                "v128_control_profile_contract": V128_CONTROL_PROFILE_CONTRACT,
                "float32_matmul_precision": execution_profile[
                    "float32_matmul_precision"
                ],
                "float32_matmul_precision_before": execution_profile[
                    "float32_matmul_precision_before"
                ],
                "ranking_reference_fraction": 0.0,
                "ranking_absolute_floor_m3": 1.0,
            }
        )
        return payload

    runner._hardware = hardware_with_v128_profile
    runner.main()


if __name__ == "__main__":
    main()
