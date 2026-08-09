from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .code_contract import rtc_source_tree_sha256
from .inp_runtime import sha256_file


GENERATION_CONTRACT = "RTC_GENERATION_KEY_V1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def generation_key(kind: str, payload: Mapping[str, object]) -> tuple[str, str]:
    """Return ``(key, code_sha)`` for one deterministic generated artefact.

    Resume is safe only when the scientific inputs *and* the implementation that generated
    them are unchanged. Binding the entire RTC Python source tree is intentionally
    conservative: a source change invalidates stale SWMM/model artefacts instead of silently
    mixing contracts inside a Fresh Workspace.
    """

    code_sha = rtc_source_tree_sha256()
    body = {
        "contract": GENERATION_CONTRACT,
        "kind": str(kind),
        "rtc_source_tree_sha256": code_sha,
        "payload": dict(payload),
    }
    key = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return key, code_sha


def hashed_file_record(path: str | Path) -> dict[str, str]:
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"generated artefact is missing: {p}")
    return {"path": str(p), "sha256": sha256_file(p)}


def verify_hashed_file(record: Mapping[str, object]) -> Path:
    p = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not p.is_file() or not expected or sha256_file(p) != expected:
        raise ValueError(f"generated artefact missing/changed: {p}")
    return p


def atomic_json_write(path: str | Path, payload: object) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
