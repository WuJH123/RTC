"""Causal Step1-state store for Project7 V127 Step2.

Historical D2/D3 branch tensors correctly use SWMM truth as future labels, but their
``initial_state`` field is the authoritative checkpoint state.  Online Step2 instead
receives a sparse-sensor Step1 estimate.  V127 removes this train/deploy mismatch by
binding each development/train event/checkpoint to the exact frozen Step1 reconstruction
computed from the same 13-frame causal sensor history used online.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np

V127_CAUSAL_STATE_CONTRACT = "PROJECT7_V127_CAUSAL_STEP1_STATE_STORE_V1"


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class CausalStateStoreV127:
    event_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    elapsed_seconds: np.ndarray
    state_si: np.ndarray
    current_setting: np.ndarray
    state_sha256: tuple[str, ...]
    step1_sha256: str
    sensor_sha256: str
    graph_sha256: str
    contract: str = V127_CAUSAL_STATE_CONTRACT

    def validate(self) -> None:
        count = len(self.event_ids)
        if self.contract != V127_CAUSAL_STATE_CONTRACT or count <= 0:
            raise ValueError("invalid/empty V127 causal Step1-state store")
        if len(self.checkpoint_ids) != count or self.elapsed_seconds.shape != (count,):
            raise ValueError("V127 causal-state identity arrays are misaligned")
        if self.state_si.ndim != 3 or self.state_si.shape[0] != count:
            raise ValueError("V127 causal state must be [checkpoint,node,state]")
        if self.current_setting.ndim != 2 or self.current_setting.shape[0] != count:
            raise ValueError("V127 current setting must be [checkpoint,actuator]")
        if not np.isfinite(self.state_si).all() or not np.isfinite(self.current_setting).all():
            raise ValueError("V127 causal state store contains non-finite values")
        if np.any((self.current_setting < -1e-7) | (self.current_setting > 1.0 + 1e-7)):
            raise ValueError("V127 causal current setting is outside [0,1]")
        keys = list(zip(self.event_ids, self.checkpoint_ids, self.elapsed_seconds.tolist(), strict=True))
        if len(set(keys)) != count:
            raise ValueError("V127 causal state store duplicates event/checkpoint identities")
        if len(self.state_sha256) != count:
            raise ValueError("V127 causal state store lacks per-state hashes")
        for i, expected in enumerate(self.state_sha256):
            if _sha256_array(self.state_si[i]) != str(expected):
                raise ValueError("V127 causal Step1 state content hash mismatch")
        for value in (self.step1_sha256, self.sensor_sha256, self.graph_sha256):
            if len(str(value)) != 64:
                raise ValueError("V127 causal state store lacks canonical frozen lineage SHA")

    def index(self) -> dict[tuple[str, str], int]:
        self.validate()
        result: dict[tuple[str, str], int] = {}
        for i, (event, checkpoint) in enumerate(zip(self.event_ids, self.checkpoint_ids, strict=True)):
            key = (str(event), str(checkpoint))
            if key in result:
                raise ValueError(f"V127 causal state duplicates event/checkpoint key: {key}")
            result[key] = i
        return result

    def state_for(self, entry) -> np.ndarray:
        key = (str(entry.event_id), str(entry.checkpoint_id))
        index = self.index().get(key)
        if index is None:
            raise KeyError(f"V127 causal state store has no {key}")
        return np.asarray(self.state_si[index], dtype=np.float32)

    def current_setting_for(self, entry) -> np.ndarray:
        key = (str(entry.event_id), str(entry.checkpoint_id))
        index = self.index().get(key)
        if index is None:
            raise KeyError(f"V127 causal state store has no {key}")
        return np.asarray(self.current_setting[index], dtype=np.float32)


def load_causal_state_store_v127(path: str | Path) -> CausalStateStoreV127:
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "contract", "event_ids", "checkpoint_ids", "elapsed_seconds", "state_si",
            "current_setting", "state_sha256", "step1_sha256", "sensor_sha256", "graph_sha256",
        }
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"V127 causal state store missing: {missing}")
        result = CausalStateStoreV127(
            event_ids=tuple(raw["event_ids"].astype(str).tolist()),
            checkpoint_ids=tuple(raw["checkpoint_ids"].astype(str).tolist()),
            elapsed_seconds=raw["elapsed_seconds"].astype(np.int64),
            state_si=raw["state_si"].astype(np.float32),
            current_setting=raw["current_setting"].astype(np.float32),
            state_sha256=tuple(raw["state_sha256"].astype(str).tolist()),
            step1_sha256=str(raw["step1_sha256"].item()),
            sensor_sha256=str(raw["sensor_sha256"].item()),
            graph_sha256=str(raw["graph_sha256"].item()),
            contract=str(raw["contract"].item()),
        )
    result.validate()
    return result


__all__ = [
    "CausalStateStoreV127",
    "V127_CAUSAL_STATE_CONTRACT",
    "load_causal_state_store_v127",
]
