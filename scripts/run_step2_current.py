"""Stable current Project7 Step2 entrypoint.

The primary research target is explicit: learn how the 109 writable facility actions change
system-wide cumulative TFV, conditioned on causal Step1 reconstructed state and causal rainfall.
The current runner keeps the validated pairwise V(candidate)-V(reference) architecture and adds a
selection-aware training stage so the model learns the HOLD=0 decision boundary needed by online
control.

Legacy V128 hydraulic-world-model training and the earlier Direct-TFV V2 runner remain in the
repository for ablation/history only.  Raw Torch Chrome tracing remains fail-closed on the 16-GB
Development workstation.
"""
from __future__ import annotations

import sys

from run_step2_tfv_value_selection_aware_current import main as _direct_main


def main() -> None:
    if (
        "--torch-profiler" in sys.argv[1:]
        and "--help" not in sys.argv[1:]
        and "-h" not in sys.argv[1:]
    ):
        raise RuntimeError(
            "current Project7 disables raw --torch-profiler trace export on the 16-GB Development "
            "workstation. Use bounded resource telemetry instead."
        )
    _direct_main()


if __name__ == "__main__":
    main()
