"""Run current Project7 stage-checkpoint audits with the training memory chunk reproduced.

The current Development audit scripts construct ``V127StreamingMemoryDesign`` internally and
historically hard-code ``hydraulic_branch_chunk=4``. A valid stage checkpoint trained with a
smaller execution-only chunk (for example ``--hydraulic-branch-chunk 2`` on an 8-GB GPU) then
fails the source-strict training-design equality check before any read-only audit can run.

This launcher fixes that interface mismatch without weakening checkpoint validation and without
changing any Step2/model/training source. It replaces only the audit module's local
``V127StreamingMemoryDesign`` constructor so the requested hydraulic branch chunk is reproduced;
the original audit then calls the unchanged source-strict ``load_stage_checkpoint_v128`` and all
other design, lineage, graph, profile, source-hash and model-state checks remain exact.

Use this only for Development stage checkpoints. It does not retrain, mutate checkpoints, enable
production runtime, or access Validation/Final/Formal/Policy-Lock data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import sys
from types import ModuleType
from typing import Callable


AUDIT_MODULES = {
    "flow": "audit_step2_actuator_flow_effect_current",
    "hydraulic": "audit_step2_direct_hydraulic_effect_current",
    "gradient": "audit_step2_gradient_stage_current_dev",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        choices=tuple(AUDIT_MODULES),
        required=True,
        help="Current Development audit to execute.",
    )
    parser.add_argument(
        "--hydraulic-branch-chunk",
        type=int,
        required=True,
        help=(
            "Execution-only hydraulic branch chunk recorded in the stage checkpoint training "
            "design. Must match the value used to create that checkpoint."
        ),
    )
    return parser


def _install_memory_design_override(
    module: ModuleType, *, hydraulic_branch_chunk: int
) -> Callable[..., object]:
    if int(hydraulic_branch_chunk) <= 0:
        raise ValueError("--hydraulic-branch-chunk must be positive")
    original = getattr(module, "V127StreamingMemoryDesign", None)
    if original is None:
        raise RuntimeError(
            f"{module.__name__} does not expose V127StreamingMemoryDesign; audit interface changed"
        )

    def compatible_memory_design(*args, **kwargs):
        # First reproduce every field exactly as the original audit requested, then change only
        # the execution-only hydraulic microbatch size. The original source-strict checkpoint
        # loader still compares the resulting complete training design against the checkpoint.
        instance = original(*args, **kwargs)
        return replace(instance, hydraulic_branch_chunk=int(hydraulic_branch_chunk))

    module.V127StreamingMemoryDesign = compatible_memory_design
    return original


def main() -> None:
    parser = _parser()
    known, remaining = parser.parse_known_args(sys.argv[1:])
    if int(known.hydraulic_branch_chunk) <= 0:
        parser.error("--hydraulic-branch-chunk must be positive")

    module_name = AUDIT_MODULES[str(known.audit)]
    module = importlib.import_module(module_name)
    original_memory_design = _install_memory_design_override(
        module, hydraulic_branch_chunk=int(known.hydraulic_branch_chunk)
    )
    previous_argv = sys.argv
    try:
        sys.argv = [str(getattr(module, "__file__", module_name)), *remaining]
        print(
            "[STAGE_CHECKPOINT_AUDIT_COMPAT] "
            f"audit={known.audit} hydraulic_branch_chunk={known.hydraulic_branch_chunk} "
            "source_strict_loader_preserved=true",
            flush=True,
        )
        module.main()
    finally:
        module.V127StreamingMemoryDesign = original_memory_design
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
