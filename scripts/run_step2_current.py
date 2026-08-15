"""Stable current Project7 Step2 entrypoint.

The caller must explicitly choose ``--profile smoke|dev|full``. This prevents an accidental
multi-hour full training run during ordinary debugging. Current V128 also installs the
mmap-backed lazy truth-branch streamer before training so large SWMM state/flow arrays are
materialized only for the active microbatch. Versioned runners remain archival/internal.

The historical ``--torch-profiler`` path records an unbounded operator trace across the whole
one-group Stage A/B0/objective sequence. On the current 16-GB Windows workstation that trace
can consume tens of GB of virtual/private memory during Chrome-trace export even after the
training stages and checkpoints have completed. The stable current surface therefore rejects
that raw trace mode before any expensive work. Use ``--profile-one-group`` without
``--torch-profiler`` and inspect ``TRAINING_TELEMETRY.jsonl`` plus external GPU sampling.
"""
from __future__ import annotations

import sys

from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming

install_v128_lazy_streaming()

from run_step2_v128_current_profiles import main as _versioned_main  # noqa: E402


def main() -> None:
    if "--torch-profiler" in sys.argv[1:] and "--help" not in sys.argv[1:] and "-h" not in sys.argv[1:]:
        raise RuntimeError(
            "current Project7 disables raw --torch-profiler trace export on the 16-GB "
            "workstation because it can exhaust host memory after successful training. "
            "Run --profile-one-group without --torch-profiler; TRAINING_TELEMETRY.jsonl "
            "is the supported low-overhead profile artifact."
        )
    _versioned_main()


if __name__ == "__main__":
    main()
