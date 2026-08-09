from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_section_payload(path: str | Path, section_name: str) -> list[str]:
    target = section_name.strip().upper()
    section = ""
    rows: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            continue
        if section != target:
            continue
        line = raw.split(";", 1)[0].strip()
        if line:
            rows.append(" ".join(line.split()))
    return rows


def section_payload_sha256(path: str | Path, section_name: str) -> str:
    rows = canonical_section_payload(path, section_name)
    if not rows:
        raise ValueError(f"section [{section_name}] contains no executable payload: {path}")
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
