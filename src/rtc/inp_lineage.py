from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_PHYSICAL_SECTIONS = {
    "JUNCTIONS",
    "OUTFALLS",
    "STORAGE",
    "DIVIDERS",
    "CONDUITS",
    "PUMPS",
    "ORIFICES",
    "WEIRS",
    "OUTLETS",
    "XSECTIONS",
    "TRANSECTS",
    "LOSSES",
    "VERTICES",
    "CURVES",
    "SUBCATCHMENTS",
    "SUBAREAS",
    "INFILTRATION",
    "LID_CONTROLS",
    "LID_USAGE",
    "AQUIFERS",
    "GROUNDWATER",
}
_FILE_TOKEN = re.compile(r"\bFILE\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_external_file(raw: str, *, inp: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = inp.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError(f"SWMM event references missing external FILE: {candidate}")
    return candidate


def _canonicalize_file_tokens(line: str, *, inp: Path) -> str:
    """Replace path spelling with external forcing content identity."""

    def replace(match: re.Match[str]) -> str:
        raw = next(group for group in match.groups() if group is not None)
        path = _resolve_external_file(raw, inp=inp)
        return f"FILE SHA256:{_sha256_file(path)}"

    return _FILE_TOKEN.sub(replace, line)


def canonical_physical_contract(path: str | Path) -> dict[str, list[str]]:
    """Return a stable forcing/control-independent physical representation of a SWMM INP."""

    sections: dict[str, list[str]] = {}
    current = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip().upper()
            continue
        if current not in _PHYSICAL_SECTIONS:
            continue
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        sections.setdefault(current, []).append(" ".join(line.split()))
    if not sections:
        raise ValueError(f"no physical SWMM sections found in {path}")
    return {name: sections[name] for name in sorted(sections)}


def physical_contract_sha256(path: str | Path) -> str:
    payload = json.dumps(
        canonical_physical_contract(path),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_scientific_event_contract(path: str | Path) -> dict[str, list[str]]:
    """Return event identity while ignoring policy/runtime-only edits.

    ``[CONTROLS]`` and ``THREADS`` are ignored. External ``FILE`` path spellings are
    replaced by the referenced file SHA-256, so relocating a runtime INP does not change
    event identity while changing the rainfall/time-series bytes does.
    """

    inp = Path(path).resolve()
    sections: dict[str, list[str]] = {}
    current = ""
    for raw in inp.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip().upper()
            continue
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if current == "CONTROLS":
            continue
        tokens = line.split()
        if current == "OPTIONS" and tokens and tokens[0].upper() == "THREADS":
            continue
        line = _canonicalize_file_tokens(line, inp=inp)
        sections.setdefault(current, []).append(" ".join(line.split()))
    if not sections:
        raise ValueError(f"no scientific SWMM event content found in {path}")
    return {name: sections[name] for name in sorted(sections)}


def external_event_file_hashes(path: str | Path) -> dict[str, str]:
    """Return auditable resolved external input paths and hashes."""

    inp = Path(path).resolve()
    current = ""
    files: dict[str, str] = {}
    for raw in inp.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip().upper()
            continue
        line = raw.split(";", 1)[0].strip()
        if not line or current == "CONTROLS":
            continue
        for match in _FILE_TOKEN.finditer(line):
            token = next(group for group in match.groups() if group is not None)
            candidate = _resolve_external_file(token, inp=inp)
            files[str(candidate)] = _sha256_file(candidate)
    return dict(sorted(files.items()))


def scientific_event_contract_sha256(path: str | Path) -> str:
    payload = json.dumps(
        canonical_scientific_event_contract(path),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_physical_contract_manifest(inp_path: str | Path, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "SWMM_PHYSICAL_NETWORK_CONTRACT_V1",
        "source_inp": str(Path(inp_path).resolve()),
        "physical_sha256": physical_contract_sha256(inp_path),
        "scientific_event_sha256": scientific_event_contract_sha256(inp_path),
        "external_event_file_sha256": external_event_file_hashes(inp_path),
        "sections": canonical_physical_contract(inp_path),
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out
