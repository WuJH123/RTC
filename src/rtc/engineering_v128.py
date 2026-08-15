"""Per-actuator engineering envelopes for Project7 V128.

The Project7 methodology testbed historically used a single [0,1] setting range and a
0.5 target change per 10 minutes for every controllable SWMM link.  That is a transparent
idealised control envelope, not evidence that heterogeneous pumps/orifices/weirs share the
same field actuation rate.

V128 makes the envelope explicit and hashable.  A real/frozen metadata file can provide
one min/max/rate triplet per ordered actuator.  When such metadata is unavailable the code
may deliberately construct the historical idealised envelope, but the source label records
that fact so it cannot be described as field engineering truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .step2_v60_contract import require_feature

V128_ENGINEERING_ENVELOPE_CONTRACT = (
    "PROJECT7_V128_PER_ACTUATOR_ENGINEERING_ENVELOPE_V1"
)
V128_IDEALIZED_ENVELOPE_SOURCE = "IDEALIZED_DEFAULT_0P5_PER_10MIN"


def _sha_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class V128EngineeringEnvelope:
    actuator_ids: tuple[str, ...]
    min_setting: np.ndarray
    max_setting: np.ndarray
    max_delta_per_10min: np.ndarray
    source: str
    source_sha256: str

    def validate(self) -> None:
        n = len(self.actuator_ids)
        if n <= 0 or len(set(self.actuator_ids)) != n:
            raise ValueError("V128 engineering envelope requires unique actuator IDs")
        lo = np.asarray(self.min_setting, dtype=float).reshape(-1)
        hi = np.asarray(self.max_setting, dtype=float).reshape(-1)
        delta = np.asarray(self.max_delta_per_10min, dtype=float).reshape(-1)
        if (len(lo), len(hi), len(delta)) != (n, n, n):
            raise ValueError("V128 engineering envelope arrays do not match actuator count")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(delta).all():
            raise ValueError("V128 engineering envelope contains non-finite values")
        if np.any(lo < 0.0) or np.any(hi > 1.0) or np.any(lo > hi):
            raise ValueError("V128 engineering setting bounds must satisfy 0 <= min <= max <= 1")
        if np.any(delta <= 0.0) or np.any(delta > 1.0):
            raise ValueError("V128 per-10min target deltas must lie in (0,1]")
        if not str(self.source).strip():
            raise ValueError("V128 engineering envelope requires a source label")
        sha = str(self.source_sha256).lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("V128 engineering envelope requires canonical source SHA256")

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update("\n".join(self.actuator_ids).encode("utf-8"))
        digest.update(str(self.source).encode("utf-8"))
        digest.update(bytes.fromhex(self.source_sha256))
        digest.update(
            bytes.fromhex(
                _sha_arrays(
                    np.asarray(self.min_setting, dtype=np.float64),
                    np.asarray(self.max_setting, dtype=np.float64),
                    np.asarray(self.max_delta_per_10min, dtype=np.float64),
                )
            )
        )
        return digest.hexdigest()

    @property
    def is_idealized_default(self) -> bool:
        return str(self.source) == V128_IDEALIZED_ENVELOPE_SOURCE

    def assert_graph_order(self, graph: Any) -> None:
        self.validate()
        if tuple(map(str, graph.actuator_ids)) != self.actuator_ids:
            raise ValueError("V128 engineering envelope actuator order differs from graph")


def idealized_engineering_envelope_v128(
    graph: Any,
    *,
    max_delta_per_10min: float = 0.5,
) -> V128EngineeringEnvelope:
    if not 0.0 < float(max_delta_per_10min) <= 1.0:
        raise ValueError("idealized V128 target delta must lie in (0,1]")
    names = tuple(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics, dtype=float)
    lo = physics[:, require_feature(names, "min_setting")].astype(np.float64)
    hi = physics[:, require_feature(names, "max_setting")].astype(np.float64)
    delta = np.full(len(graph.actuator_ids), float(max_delta_per_10min), dtype=np.float64)
    source_payload = {
        "contract": V128_ENGINEERING_ENVELOPE_CONTRACT,
        "source": V128_IDEALIZED_ENVELOPE_SOURCE,
        "max_delta_per_10min": float(max_delta_per_10min),
        "actuator_ids": list(map(str, graph.actuator_ids)),
        "min_setting": lo.tolist(),
        "max_setting": hi.tolist(),
    }
    source_sha = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    envelope = V128EngineeringEnvelope(
        actuator_ids=tuple(map(str, graph.actuator_ids)),
        min_setting=lo,
        max_setting=hi,
        max_delta_per_10min=delta,
        source=V128_IDEALIZED_ENVELOPE_SOURCE,
        source_sha256=source_sha,
    )
    envelope.validate()
    return envelope


def load_engineering_envelope_v128(path: str | Path, *, graph: Any) -> V128EngineeringEnvelope:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("contract") != V128_ENGINEERING_ENVELOPE_CONTRACT:
        raise ValueError("not a current V128 engineering-envelope file")
    rows = payload.get("actuators")
    if not isinstance(rows, list) or not rows:
        raise ValueError("V128 engineering envelope requires an actuator row list")
    ids: list[str] = []
    lo: list[float] = []
    hi: list[float] = []
    delta: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("V128 engineering actuator row must be an object")
        ids.append(str(row.get("actuator_id", "")))
        try:
            lo.append(float(row["min_setting"]))
            hi.append(float(row["max_setting"]))
            delta.append(float(row["max_delta_per_10min"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("V128 engineering actuator row lacks valid numeric bounds/rate") from exc
    envelope = V128EngineeringEnvelope(
        actuator_ids=tuple(ids),
        min_setting=np.asarray(lo, dtype=np.float64),
        max_setting=np.asarray(hi, dtype=np.float64),
        max_delta_per_10min=np.asarray(delta, dtype=np.float64),
        source=str(payload.get("source", source.name)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    envelope.assert_graph_order(graph)
    return envelope


def save_idealized_engineering_envelope_v128(graph: Any, path: str | Path) -> Path:
    envelope = idealized_engineering_envelope_v128(graph)
    rows = [
        {
            "actuator_id": aid,
            "min_setting": float(lo),
            "max_setting": float(hi),
            "max_delta_per_10min": float(delta),
        }
        for aid, lo, hi, delta in zip(
            envelope.actuator_ids,
            envelope.min_setting,
            envelope.max_setting,
            envelope.max_delta_per_10min,
            strict=True,
        )
    ]
    payload = {
        "contract": V128_ENGINEERING_ENVELOPE_CONTRACT,
        "source": V128_IDEALIZED_ENVELOPE_SOURCE,
        "field_engineering_claim": False,
        "actuators": rows,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


__all__ = [
    "V128_ENGINEERING_ENVELOPE_CONTRACT",
    "V128_IDEALIZED_ENVELOPE_SOURCE",
    "V128EngineeringEnvelope",
    "idealized_engineering_envelope_v128",
    "load_engineering_envelope_v128",
    "save_idealized_engineering_envelope_v128",
]
