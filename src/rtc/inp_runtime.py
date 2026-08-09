from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_FILE_TOKEN = re.compile(r"\bFILE\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))", re.IGNORECASE)


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


def _preserve_external_file_reference(line: str, *, source_dir: Path) -> str:
    """Rewrite relative SWMM ``FILE`` inputs to absolute paths before relocating the INP."""

    body, sep, comment = line.partition(";")

    def replace(match: re.Match[str]) -> str:
        raw = next(group for group in match.groups() if group is not None)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = source_dir / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError(f"SWMM INP references missing external FILE: {candidate}")
        # Quoting makes spaces safe and is accepted by SWMM FILE references.
        return f'FILE "{candidate}"'

    rewritten = _FILE_TOKEN.sub(replace, body)
    return rewritten + ((";" + comment) if sep else "")


def build_runtime_inp(
    source: str | Path,
    destination: str | Path,
    *,
    native_controls: bool,
    swmm_threads: int | None = None,
) -> RuntimeInpContract:
    """Create a policy-isolated runtime INP without changing event forcing or hydraulics.

    ``native_controls=False`` removes only executable lines inside ``[CONTROLS]``.
    ``THREADS`` may be changed as an execution-only option. If the source INP references
    external rainfall/time-series files with relative ``FILE`` paths, they are rewritten to
    absolute paths before the runtime INP is relocated, preserving the exact forcing bytes.
    """

    src = Path(source).resolve()
    dst = Path(destination)
    if swmm_threads is not None and swmm_threads <= 0:
        raise ValueError("swmm_threads must be positive or None")

    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    output: list[str] = []
    section = ""
    threads_seen = False
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        core = line[:-1] if newline else line
        stripped = core.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            output.append(core + newline)
            continue

        if section == "CONTROLS" and not native_controls:
            if not core.split(";", 1)[0].strip():
                output.append(core + newline)
            continue

        if section != "CONTROLS" and "FILE" in core.upper():
            core = _preserve_external_file_reference(core, source_dir=src.parent)

        if section == "OPTIONS" and swmm_threads is not None:
            body = core.split(";", 1)[0].strip().split()
            if body and body[0].upper() == "THREADS":
                core = _replace_option(core, key="THREADS", value=str(int(swmm_threads)))
                threads_seen = True
        output.append(core + newline)

    if swmm_threads is not None and not threads_seen:
        raise ValueError("[OPTIONS] THREADS was not found; refuse to silently mutate layout")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(output), encoding="utf-8")
    if not native_controls:
        assert_native_controls_disabled(dst)
    return RuntimeInpContract(
        source_path=str(src),
        runtime_path=str(dst.resolve()),
        source_sha256=sha256_file(src),
        runtime_sha256=sha256_file(dst),
        native_controls_enabled=bool(native_controls),
        swmm_threads=swmm_threads,
    )
