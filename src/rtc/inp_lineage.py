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
        normalized = " ".join(line.split())
        sections.setdefault(current, []).append(normalized)
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
    """Return complete event INP content while ignoring policy/runtime-only edits."""

    sections: dict[str, list[str]] = {}
    current = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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
        sections.setdefault(current, []).append(" ".join(tokens))
    if not sections:
        raise ValueError(f"no scientific SWMM event content found in {path}")
    return {name: sections[name] for name in sorted(sections)}


def external_event_file_hashes(path: str | Path) -> dict[str, str]:
    """Hash external files referenced by ``FILE`` tokens in the scientific INP.

    Rain gages/time series can be embedded in the INP or delegated to external files. The
    event identity must follow those bytes as well, otherwise changing an external rainfall
    file in place would leave an apparently unchanged Final event contract.
    """

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
            candidate = Path(token).expanduser()
            if not candidate.is_absolute():
                candidate = inp.parent / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise ValueError(f"SWMM event references missing external FILE: {candidate}")
            try:
                key = candidate.relative_to(inp.parent).as_posix()
            except ValueError:
                key = str(candidate)
            files[key] = _sha256_file(candidate)
    return dict(sorted(files.items()))


def scientific_event_contract_sha256(path: str | Path) -> str:
    payload = json.dumps(
        {
            "inp": canonical_scientific_event_contract(path),
            "external_files": external_event_file_hashes(path),
        },
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
