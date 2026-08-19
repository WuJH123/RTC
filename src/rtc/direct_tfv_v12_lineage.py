"""Behavioral lineage for the Development V12 causal rainfall scenario-mean query."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .checkpoint_direct_tfv import direct_tfv_first_move_behavioral_source_sha256
from .step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


V12_RAINFALL_MULTIPLIERS = (0.8, 1.0, 1.2)
V12_RAINFALL_HISTORY_STEPS = 3
V12_RAINFALL_DECAY_PER_STEP = 0.92
V12_RAINFALL_AGGREGATION = "MEAN_PREDICTED_DELTA_TFV"


def direct_tfv_v12_behavioral_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    digest.update(direct_tfv_first_move_behavioral_source_sha256().encode("utf-8"))
    digest.update(hashlib.sha256((root / "step3_tfv_value_mpc_v10.py").read_bytes()).digest())
    for value in (
        DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
        repr(V12_RAINFALL_MULTIPLIERS),
        str(V12_RAINFALL_HISTORY_STEPS),
        str(V12_RAINFALL_DECAY_PER_STEP),
        V12_RAINFALL_AGGREGATION,
    ):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "V12_RAINFALL_AGGREGATION",
    "V12_RAINFALL_DECAY_PER_STEP",
    "V12_RAINFALL_HISTORY_STEPS",
    "V12_RAINFALL_MULTIPLIERS",
    "direct_tfv_v12_behavioral_sha256",
]
