"""Run the single bounded V7 historical joint-action interaction follow-up.

This wrapper reuses the current Direct-TFV core runner unchanged for data, split, normalization,
evaluation, checkpointing and reporting.  It swaps only the model class and trainer.  The frozen
V23/V5 publication policy remains untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_step2_tfv_value_core_current as core

from rtc.step2_tfv_value_historical_v7 import (
    HISTORICAL_INTERACTION_VALUE_CONTRACT,
    HistoricalInteractionTFVValueModelV7,
)
from rtc.step2_tfv_value_training_historical_v7 import (
    HISTORICAL_INTERACTION_TRAINING_CONTRACT,
    train_direct_tfv_value_model_historical_v7,
)


HISTORICAL_INTERACTION_RUN_CONTRACT = (
    "PROJECT7_DIRECT_TFV_HISTORICAL_ACTION_IDENTITY_INTERACTION_V7_DEV"
)


def main() -> None:
    core.CURRENT_DIRECT_TFV_RUN_CONTRACT = HISTORICAL_INTERACTION_RUN_CONTRACT
    core.DIRECT_TFV_VALUE_CONTRACT = HISTORICAL_INTERACTION_VALUE_CONTRACT
    core.DIRECT_TFV_TRAINING_CONTRACT = HISTORICAL_INTERACTION_TRAINING_CONTRACT
    core.DirectFacilityTFVValueModel = HistoricalInteractionTFVValueModelV7
    core.train_direct_tfv_value_model_v4 = train_direct_tfv_value_model_historical_v7
    core.main()


if __name__ == "__main__":
    main()
