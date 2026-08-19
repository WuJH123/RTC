"""Candidate-free causal context store for Direct-TFV first-move calibration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT = (
    "PROJECT7_DIRECT_TFV_FIRST_MOVE_CANDIDATE_FREE_CONTEXT_STORE_V1"
)


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class FirstMoveCalibrationContextStore:
    event_ids: tuple[str, ...]
    rainfall_groups: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    elapsed_seconds: np.ndarray
    inp_paths: tuple[str, ...]
    trajectory_metadata_paths: tuple[str, ...]
    scientific_splits: tuple[str, ...]
    development_folds: tuple[str, ...]
    current_state: np.ndarray
    rainfall_history: np.ndarray
    default_rainfall_forecast: np.ndarray
    active_target: np.ndarray
    previous_actuator_flow: np.ndarray
    prefix_sha256: tuple[str, ...]
    context_sha256: tuple[str, ...]
    graph_sha256: str
    step1_sha256: str
    sensors_sha256: str
    candidate_rows_used: bool = False
    generic_d3_candidate_dependency: bool = False
    causal_future_rainfall_used: bool = False
    contract: str = DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT

    def validate(self) -> None:
        n = len(self.event_ids)
        if self.contract != DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT or n <= 0:
            raise ValueError("invalid/empty Direct-TFV first-move context store")
        if self.candidate_rows_used or self.generic_d3_candidate_dependency:
            raise ValueError("first-move context must be independent of generic D3 candidate rows")
        if self.causal_future_rainfall_used:
            raise ValueError("first-move context store indicates future rainfall leakage")
        for values in (
            self.rainfall_groups, self.checkpoint_ids, self.inp_paths,
            self.trajectory_metadata_paths, self.scientific_splits, self.development_folds,
            self.prefix_sha256, self.context_sha256,
        ):
            if len(values) != n:
                raise ValueError("first-move context identity arrays are misaligned")
        if self.elapsed_seconds.shape != (n,):
            raise ValueError("first-move elapsed array is misaligned")
        if self.current_state.ndim != 3 or self.current_state.shape[0] != n:
            raise ValueError("first-move current_state must be [context,node,state]")
        if self.rainfall_history.ndim != 4 or self.rainfall_history.shape[0] != n:
            raise ValueError("first-move rainfall_history must be [context,history,node,feature]")
        if self.default_rainfall_forecast.ndim != 5 or self.default_rainfall_forecast.shape[0] != n:
            raise ValueError("default rainfall forecast must be [context,scenario,H,node,feature]")
        if self.active_target.shape != (n, 109) or self.previous_actuator_flow.shape != (n, 109):
            raise ValueError("first-move context target/flow arrays must be [context,109]")
        for value in (
            self.current_state, self.rainfall_history, self.default_rainfall_forecast,
            self.active_target, self.previous_actuator_flow,
        ):
            if not np.isfinite(value).all():
                raise ValueError("first-move context contains non-finite values")
        if np.any((self.active_target < -1e-7) | (self.active_target > 1.0 + 1e-7)):
            raise ValueError("first-move active target leaves [0,1]")
        keys = list(zip(self.event_ids, self.rainfall_groups, self.checkpoint_ids, strict=True))
        if len(set(keys)) != n:
            raise ValueError("first-move context duplicates event/rainfall/checkpoint identity")
        for i in range(n):
            digest = hashlib.sha256()
            for array in (
                self.current_state[i], self.rainfall_history[i], self.default_rainfall_forecast[i],
                self.active_target[i], self.previous_actuator_flow[i],
            ):
                digest.update(_sha_array(array).encode("utf-8"))
            if digest.hexdigest() != self.context_sha256[i]:
                raise ValueError("first-move context content hash mismatch")
        for value in (self.graph_sha256, self.step1_sha256, self.sensors_sha256, *self.prefix_sha256):
            if len(str(value)) != 64:
                raise ValueError("first-move context lacks canonical lineage SHA")

    def index(self) -> dict[str, int]:
        self.validate()
        return {group: i for i, group in enumerate(self.rainfall_groups)}

    def entry(self, rainfall_group: str) -> dict[str, Any]:
        i = self.index().get(str(rainfall_group))
        if i is None:
            raise KeyError(f"first-move context has no rainfall group {rainfall_group!r}")
        return {
            "event_id": self.event_ids[i],
            "rainfall_group": self.rainfall_groups[i],
            "checkpoint_id": self.checkpoint_ids[i],
            "elapsed_seconds": int(self.elapsed_seconds[i]),
            "inp_path": self.inp_paths[i],
            "trajectory_metadata_path": self.trajectory_metadata_paths[i],
            "scientific_split": self.scientific_splits[i],
            "development_fold": self.development_folds[i],
            "current_state": self.current_state[i],
            "rainfall_history": self.rainfall_history[i],
            "default_rainfall_forecast": self.default_rainfall_forecast[i],
            "active_target": self.active_target[i],
            "previous_actuator_flow": self.previous_actuator_flow[i],
            "prefix_sha256": self.prefix_sha256[i],
            "context_sha256": self.context_sha256[i],
        }


def load_first_move_context_store(path: str | Path) -> FirstMoveCalibrationContextStore:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT:
        raise ValueError("wrong first-move context manifest contract")
    data_path = Path(str(payload.get("data_path", "")))
    if not data_path.is_absolute():
        data_path = (manifest_path.parent / data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"first-move context data missing: {data_path}")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != str(payload.get("data_sha256", "")):
        raise ValueError("first-move context data SHA mismatch")
    with np.load(data_path, allow_pickle=False) as raw:
        result = FirstMoveCalibrationContextStore(
            event_ids=tuple(raw["event_ids"].astype(str).tolist()),
            rainfall_groups=tuple(raw["rainfall_groups"].astype(str).tolist()),
            checkpoint_ids=tuple(raw["checkpoint_ids"].astype(str).tolist()),
            elapsed_seconds=raw["elapsed_seconds"].astype(np.int64),
            inp_paths=tuple(raw["inp_paths"].astype(str).tolist()),
            trajectory_metadata_paths=tuple(raw["trajectory_metadata_paths"].astype(str).tolist()),
            scientific_splits=tuple(raw["scientific_splits"].astype(str).tolist()),
            development_folds=tuple(raw["development_folds"].astype(str).tolist()),
            current_state=raw["current_state"].astype(np.float32),
            rainfall_history=raw["rainfall_history"].astype(np.float32),
            default_rainfall_forecast=raw["default_rainfall_forecast"].astype(np.float32),
            active_target=raw["active_target"].astype(np.float32),
            previous_actuator_flow=raw["previous_actuator_flow"].astype(np.float32),
            prefix_sha256=tuple(raw["prefix_sha256"].astype(str).tolist()),
            context_sha256=tuple(raw["context_sha256"].astype(str).tolist()),
            graph_sha256=str(raw["graph_sha256"].item()),
            step1_sha256=str(raw["step1_sha256"].item()),
            sensors_sha256=str(raw["sensors_sha256"].item()),
            candidate_rows_used=bool(raw["candidate_rows_used"].item()),
            generic_d3_candidate_dependency=bool(raw["generic_d3_candidate_dependency"].item()),
            causal_future_rainfall_used=bool(raw["causal_future_rainfall_used"].item()),
            contract=str(raw["contract"].item()),
        )
    result.validate()
    return result


__all__ = [
    "DIRECT_TFV_FIRST_MOVE_CONTEXT_CONTRACT",
    "FirstMoveCalibrationContextStore",
    "load_first_move_context_store",
]
