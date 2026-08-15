"""Stable current Project7 Step2 entrypoint.

The caller must explicitly choose ``--profile smoke|dev|full``. This prevents an accidental
multi-hour full training run during ordinary debugging. Current V128 also installs the
mmap-backed lazy truth-branch streamer before training so large SWMM state/flow arrays are
materialized only for the active microbatch. Versioned runners remain archival/internal.
"""
from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming

install_v128_lazy_streaming()

from run_step2_v128_current_profiles import main  # noqa: E402


if __name__ == "__main__":
    main()
