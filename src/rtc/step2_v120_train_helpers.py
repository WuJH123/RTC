"""Shared non-training helpers for the one canonical V120 trainer."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphSchema
from .step2_v120_data_contract import (
    finite_auxiliary_value_metrics,
    validate_canonical_cache_population,
    validate_internal_holdout_fraction,
    verify_d2_source_audit,
)

FROZEN_SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"


def load_graph_v120(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def sha256_file_v120(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head_v120() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def value_gate_v120(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = {"rank": 0.35, "pairwise": 0.60, "sign_accuracy": 0.60, "top1_rate": 0.25}
    reasons: list[str] = []
    for key, floor in rules.items():
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or value < floor:
            reasons.append(f"{key}={value} < {floor}")
    for key in ("spread_ratio", "response_ratio"):
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value) or not 0.25 <= value <= 2.50:
            reasons.append(f"{key}={value} outside [0.25,2.50]")
    if not np.isfinite(float(metrics.get("mean_regret_m3", float("nan")))):
        reasons.append("mean_regret_m3 is non-finite")
    return not reasons, reasons


def branch_count_v120(cache: Any, names: list[str]) -> int:
    return int(sum(len(cache.entry(name).indices) for name in names))


def candidate_count_v120(cache: Any, names: list[str]) -> int:
    return int(sum(max(len(cache.entry(name).indices) - 1, 0) for name in names))


def load_frozen_train_events_v120(path: str | Path) -> tuple[dict[str, Any], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != FROZEN_SPLIT_CONTRACT:
        raise ValueError("V120 requires the frozen Project7 18/6/6 split contract")
    train = {str(x) for x in payload.get("development_train", [])}
    validation = {str(x) for x in payload.get("development_validation", [])}
    final = {str(x) for x in payload.get("final", [])}
    if (len(train), len(validation), len(final)) != (18, 6, 6):
        raise ValueError("V120 split contract must remain 18 Train / 6 Validation / 6 Final")
    if train & validation or train & final or validation & final:
        raise ValueError("V120 frozen scientific event splits overlap")
    return payload, train


__all__ = [
    "FROZEN_SPLIT_CONTRACT",
    "branch_count_v120",
    "candidate_count_v120",
    "finite_auxiliary_value_metrics",
    "git_head_v120",
    "load_frozen_train_events_v120",
    "load_graph_v120",
    "sha256_file_v120",
    "validate_canonical_cache_population",
    "validate_internal_holdout_fraction",
    "value_gate_v120",
    "verify_d2_source_audit",
]
