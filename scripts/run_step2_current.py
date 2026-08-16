"""Stable current Project7 Step2 entrypoint.

The current smoke/dev surface keeps the counterfactual-first V128 surrogate and uses balanced,
FIT-only same-prefix direct supervision in Stage A.  D2 direct pairs are prioritised for non-zero
authoritative response, actuator coverage and response magnitude before B0 is considered.
``--profile full`` remains intentionally blocked until Development direct-flow, strict q-only
hydraulic, trajectory/ranking and closed-loop gates pass.

Raw Torch Chrome tracing remains disabled on the 16-GB Windows workstation because its trace
export can consume tens of GB of virtual/private memory after successful training.  Use normal
TRAINING_TELEMETRY.jsonl and external low-frequency GPU sampling instead.
"""
from __future__ import annotations

import sys

from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming

install_v128_lazy_streaming()

from run_step2_balanced_stagea_current import main as _current_main  # noqa: E402


def main() -> None:
    if (
        "--torch-profiler" in sys.argv[1:]
        and "--help" not in sys.argv[1:]
        and "-h" not in sys.argv[1:]
    ):
        raise RuntimeError(
            "current Project7 disables raw --torch-profiler trace export on the 16-GB "
            "workstation because it can exhaust host memory after successful training. "
            "Use TRAINING_TELEMETRY.jsonl plus external low-frequency GPU sampling."
        )
    _current_main()


if __name__ == "__main__":
    main()
