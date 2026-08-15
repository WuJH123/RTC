"""Stable current Project7 Step2 entrypoint.

The caller must explicitly choose ``--profile smoke|dev|full``.  This prevents an accidental
multi-hour full training run during ordinary debugging.  The entrypoint also activates the
canonical float32 pair-census fix before importing the staged runner, so current training
cannot reproduce the historical 544/542 coverage mismatch.
"""
from rtc.step2_train_v128_current import activate_current_truth_partition

activate_current_truth_partition()

from run_step2_v128_current_profiles import main  # noqa: E402


if __name__ == "__main__":
    main()
