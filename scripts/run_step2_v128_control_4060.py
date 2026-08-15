"""Train the V128 typed actuator-message Step2 on the canonical streaming pipeline.

The large memory-safe data/training loop remains single-sourced in
``run_step2_v127_control_streaming.py``.  V128 substitutes only explicit versioned hooks:

* typed/physics-aware actuator-to-node message architecture;
* contract-strict V128 checkpoint saver;
* 1 m3 absolute action-ranking floor (no event-size proportional deadband);
* RTX-4060 FP32 matmul execution profile.

The wrapper also redirects output names so a V128 checkpoint/report can never be mistaken
for V127.  No SWMM labels, causal splits, horizons or authoritative truth are changed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from rtc.checkpoint_v128 import save_step2_v128
from rtc.step2_differentiable_v128 import (
    V128_STEP2_CONTRACT,
    build_v128_model_from_graph,
)
from rtc.v128_control_profile import (
    V128_CONTROL_PROFILE_CONTRACT,
    build_v128_control_training_design,
    configure_v128_cuda_matmul_precision,
)

V128_STREAMING_RUN_CONTRACT = (
    "PROJECT7_V128_TYPED_ACTUATOR_CONTROL_IDENTIFIABILITY_4060_STREAMING_V2"
)
V128_REPORT_FILENAME = "STEP2_V128_CONTROL_BASE_REPORT.json"
V128_CHECKPOINT_FILENAME = "step2_v128_control_base.pt"


def _load_v127_runner() -> ModuleType:
    path = Path(__file__).with_name("run_step2_v127_control_streaming.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_streaming_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 streaming runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_out_dir() -> Path:
    try:
        index = sys.argv.index("--out-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("V128 runner requires an explicit --out-dir") from exc


def main() -> None:
    execution_profile = configure_v128_cuda_matmul_precision()
    runner = _load_v127_runner()
    out_dir = _cli_out_dir()

    # Keep one canonical implementation of group streaming, split discipline and losses.
    # The substituted hooks are all contract-versioned and covered by strict checkpoint
    # loading, so V127 and V128 artifacts cannot be interchanged accidentally.
    runner.V127ControlTrainingDesign = build_v128_control_training_design
    runner.build_v127_model_from_graph = build_v128_model_from_graph
    runner.V127_STREAMING_RUN_CONTRACT = V128_STREAMING_RUN_CONTRACT

    def save_v128_redirect(path, **kwargs):
        del path
        return save_step2_v128(out_dir / V128_CHECKPOINT_FILENAME, **kwargs)

    runner.save_step2_v127 = save_v128_redirect

    base_hardware = runner._hardware

    def hardware_with_v128_profile(device):
        payload = dict(base_hardware(device))
        payload.update(
            {
                "v128_control_profile_contract": V128_CONTROL_PROFILE_CONTRACT,
                "v128_step2_contract": V128_STEP2_CONTRACT,
                "typed_physics_aware_actuator_messages": True,
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

    historical_report = out_dir / "STEP2_V127_CONTROL_BASE_REPORT.json"
    v128_report = out_dir / V128_REPORT_FILENAME
    if not historical_report.is_file():
        raise RuntimeError(
            "canonical streaming runner did not emit its expected intermediate report"
        )
    if v128_report.exists():
        raise RuntimeError(f"refusing to overwrite existing V128 report: {v128_report}")
    historical_report.replace(v128_report)
    if not (out_dir / V128_CHECKPOINT_FILENAME).is_file():
        raise RuntimeError("V128 strict checkpoint was not created")


if __name__ == "__main__":
    main()
