from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Sections that define the drainage system's physical hydraulic asset contract. Controls,
# rainfall/event forcing, reporting, coordinates and simulation dates are intentionally not
# part of this fingerprint. This allows event-specific forcing INPs and passive No-RTC
# variants to be compared while still proving that the same physical network was used.
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
    """Return the complete scientific event identity while ignoring policy/runtime-only edits.

    The runtime builders are allowed to change only executable ``[CONTROLS]`` and the
    ``THREADS`` execution option. Everything else -- rainfall/time-series forcing, dates,
    hydraulic options, network, runoff parameters and routing configuration -- belongs to
    the event identity. Consequently No-control/Internal/Proposed variants of the *same*
    event share this contract, while a different rainfall event cannot be mislabeled as the
    same Formal event merely because the physical network is unchanged.
    """

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
        normalized = " ".join(tokens)
        sections.setdefault(current, []).append(normalized)
    if not sections:
        raise ValueError(f"no scientific SWMM event content found in {path}")
    return {name: sections[name] for name in sorted(sections)}


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
        "sections": canonical_physical_contract(inp_path),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
