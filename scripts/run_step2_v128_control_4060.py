"""Train the V128 typed actuator-message Step2 on the canonical streaming pipeline.

The large memory-safe data/split loop remains single-sourced in
``run_step2_v127_control_streaming.py``. V128 substitutes every component whose semantics
change: typed architecture, typed Stage-A teacher forcing, exact full within-group pairwise
first-order gradients, strict checkpoint identity and the RTX-4060 execution profile.
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
from rtc.step2_train_v128_exact import (
    V128_OBJECTIVE_TRAINING_CONTRACT,
    train_objective_stage_streaming_v128,
)
from rtc.step2_train_v128_hydraulic import (
    V128_HYDRAULIC_TRAINING_CONTRACT,
    train_hydraulic_stage_streaming_v128,
)
from rtc.v128_control_profile import (
    V128_CONTROL_PROFILE_CONTRACT,
    build_v128_control_training_design,
    configure_v128_cuda_matmul_precision,
)

V128_STREAMING_RUN_CONTRACT = (
    "PROJECT7_V128_TYPED_ACTUATOR_EXACT_PAIRWISE_4060_STREAMING_V6_CURRENT_CLI"
)
V128_REPORT_FILENAME = "STEP2_V128_CONTROL_BASE_REPORT.json"
V128_CHECKPOINT_FILENAME = "step2_v128_control_base.pt"


def _load_v127_runner() -> ModuleType:
    """Load the audited streaming orchestration only; user-facing routing never targets V127."""
    path = Path(__file__).with_name("run_step2_v127_control_streaming.py")
    spec = importlib.util.spec_from_file_location("_rtc_shared_streaming_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared streaming orchestration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _help_requested() -> bool:
    return any(arg in {"-h", "--help"} for arg in sys.argv[1:])


def _cli_out_dir() -> Path:
    try:
        index = sys.argv.index("--out-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("V128 runner requires an explicit --out-dir") from exc


def main() -> None:
    runner = _load_v127_runner()

    # Help must be owned by argparse.  The previous wrapper tried to extract --out-dir
    # before the shared parser saw --help, so the canonical current entrypoint failed its
    # own CLI preflight.  Reuse the audited parser without starting training or touching
    # CUDA execution configuration.
    if _help_requested():
        runner.__doc__ = __doc__
        runner.main()
        return

    execution_profile = configure_v128_cuda_matmul_precision()
    out_dir = _cli_out_dir()

    # The shared runner owns only unchanged data/split/memory orchestration. Every semantic
    # component of current V128 training is replaced explicitly here and regression-tested.
    runner.V127ControlTrainingDesign = build_v128_control_training_design
    runner.build_v127_model_from_graph = build_v128_model_from_graph
    runner.train_hydraulic_stage_streaming_v127 = train_hydraulic_stage_streaming_v128
    runner.train_objective_stage_streaming_v127 = train_objective_stage_streaming_v128
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
                "v128_hydraulic_training_contract": V128_HYDRAULIC_TRAINING_CONTRACT,
                "v128_objective_training_contract": V128_OBJECTIVE_TRAINING_CONTRACT,
                "typed_action_context_used_in_teacher_forcing": True,
                "typed_physics_aware_actuator_messages": True,
                "exact_two_pass_full_pairwise_first_order_gradient": True,
                "cross_microbatch_candidate_ranking": True,
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
            "shared streaming orchestration did not emit its expected intermediate report"
        )
    if v128_report.exists():
        raise RuntimeError(f"refusing to overwrite existing V128 report: {v128_report}")
    historical_report.replace(v128_report)
    if not (out_dir / V128_CHECKPOINT_FILENAME).is_file():
        raise RuntimeError("V128 strict checkpoint was not created")


if __name__ == "__main__":
    main()
