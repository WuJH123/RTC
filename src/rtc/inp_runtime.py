from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeInpContract:
    source_path: str
    runtime_path: str
    source_sha256: str
    runtime_sha256: str
    native_controls_enabled: bool
    swmm_threads: int | None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def section_has_payload(path: str | Path, section_name: str) -> bool:
    target = section_name.strip().upper()
    section = ""
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            continue
        if section == target:
            value = raw.split(";", 1)[0].strip()
            if value:
                return True
    return False


def assert_native_controls_disabled(path: str | Path) -> None:
    if section_has_payload(path, "CONTROLS"):
        raise ValueError(
            "Python-controlled/D1/D2/D3 runs require an INP with native [CONTROLS] disabled. "
            "Use build_runtime_inp(..., native_controls=False). Internal-RTC is evaluated "
            "separately on the original frozen INP."
        )


def _replace_option(line: str, *, key: str, value: str) -> str:
    body, sep, comment = line.partition(";")
    tokens = body.split()
    if tokens and tokens[0].upper() == key.upper():
        prefix = f"{key:<20}{value}"
        return prefix + ((" ;" + comment) if sep else "") + ("\n" if line.endswith("\n") else "")
    return line


def build_runtime_inp(
    source: str | Path,
    destination: str | Path,
    *,
    native_controls: bool,
    swmm_threads: int | None = None,
) -> RuntimeInpContract:
    """Create a policy-isolated runtime INP without changing physical hydraulics.

    ``native_controls=False`` removes only executable lines inside ``[CONTROLS]``. This is
    the required base for Proposed, No-control, Hold, diagnostics and D1/D2/D3 data. The
    original frozen INP is reserved for the Internal-RTC baseline. ``swmm_threads`` changes
    only the engine execution option and is useful when running many independent SWMM
    processes in parallel; use 1 per process to avoid CPU oversubscription.
    """

    src = Path(source)
    dst = Path(destination)
    if swmm_threads is not None and swmm_threads <= 0:
        raise ValueError("swmm_threads must be positive or None")

    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    output: list[str] = []
    section = ""
    threads_seen = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            output.append(line)
            continue

        if section == "CONTROLS" and not native_controls:
            # Preserve blank/comment lines for readability but remove every executable rule.
            if not line.split(";", 1)[0].strip():
                output.append(line)
            continue

        if section == "OPTIONS" and swmm_threads is not None:
            body = line.split(";", 1)[0].strip().split()
            if body and body[0].upper() == "THREADS":
                line = _replace_option(line, key="THREADS", value=str(int(swmm_threads)))
                threads_seen = True
        output.append(line)

    if swmm_threads is not None and not threads_seen:
        raise ValueError("[OPTIONS] THREADS was not found; refuse to silently mutate layout")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(output), encoding="utf-8")
    if not native_controls:
        assert_native_controls_disabled(dst)
    return RuntimeInpContract(
        source_path=str(src.resolve()),
        runtime_path=str(dst.resolve()),
        source_sha256=sha256_file(src),
        runtime_sha256=sha256_file(dst),
        native_controls_enabled=bool(native_controls),
        swmm_threads=swmm_threads,
    )
