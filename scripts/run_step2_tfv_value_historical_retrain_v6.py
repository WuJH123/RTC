"""Run the development-only historical-preservation Direct-TFV Step2 arm.

This wrapper deliberately reuses the current core runner's exact data lineage, deterministic split,
normalization, evaluation and checkpoint format.  Only the trainer is replaced.  The frozen V23/V5
runtime and Formal lineage are not modified.
"""
from __future__ import annotations

import scripts.run_step2_tfv_value_core_current as core

from rtc.step2_tfv_value_training_historical_v6 import (
    HISTORICAL_RETRAIN_CONTRACT,
    train_direct_tfv_value_model_historical_v6,
)


HISTORICAL_RETRAIN_RUN_CONTRACT = "PROJECT7_DIRECT_TFV_HISTORICAL_PRESERVE_MAIN_RETRAIN_V6_DEV"


def main() -> None:
    core.CURRENT_DIRECT_TFV_RUN_CONTRACT = HISTORICAL_RETRAIN_RUN_CONTRACT
    core.DIRECT_TFV_TRAINING_CONTRACT = HISTORICAL_RETRAIN_CONTRACT
    core.train_direct_tfv_value_model_v4 = train_direct_tfv_value_model_historical_v6
    core.main()


if __name__ == "__main__":
    main()
