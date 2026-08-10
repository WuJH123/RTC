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
    native_controls_template_path: str | None = None
    native_controls_template_sha256: str | None = None


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
            "separately with event forcing plus the frozen native-controls template."
        )


def assert_native_controls_enabled(path: str | Path) -> None:
    if not section_has_payload(path, "CONTROLS"):
        raise ValueError(
            "native_controls=True produced no executable [CONTROLS]. Internal-RTC must use "
            "the same event forcing/DWF/initial conditions as the paired event plus a verified "
            "frozen native-controls template."
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
        return f'FILE "{candidate}"'

    rewritten = _FILE_TOKEN.sub(replace, body)
    return rewritten + ((";" + comment) if sep else "")


def _section_body_lines(path: str | Path, section_name: str) -> list[str]:
    target = section_name.strip().upper()
    section = ""
    body: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            continue
        if section == target:
            body.append(raw)
    return body


def _replace_section_body(
    lines: list[str], *, section_name: str, replacement: list[str]
) -> list[str]:
    target = section_name.strip().upper()
    header_index: int | None = None
    next_header: int | None = None
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        section = stripped[1:-1].strip().upper()
        if header_index is None and section == target:
            header_index = i
            continue
        if header_index is not None:
            next_header = i
            break
    if header_index is None:
        prefix = list(lines)
        if prefix and prefix[-1].strip():
            prefix.append("\n")
        prefix.append(f"[{target}]\n")
        prefix.extend(replacement)
        return prefix
    stop = len(lines) if next_header is None else next_header
    return [*lines[: header_index + 1], *replacement, *lines[stop:]]


def _assert_controls_template_compatible(event_source: Path, template: Path) -> None:
    """Require identical hydraulic node/actuator identities before copying native controls."""

    from .inp import discover_actuators, discover_nodes

    event_nodes = tuple(discover_nodes(event_source))
    template_nodes = tuple(discover_nodes(template))
    if event_nodes != template_nodes:
        raise ValueError(
            "native-controls template node ordering/identity differs from the event INP; "
            "refuse to splice controls across different networks"
        )
    event_catalog = discover_actuators(event_source)
    template_catalog = discover_actuators(template)
    event_signature = tuple(
        (a.actuator_id, a.kind, a.upstream_node, a.downstream_node)
        for a in event_catalog.actuators
    )
    template_signature = tuple(
        (a.actuator_id, a.kind, a.upstream_node, a.downstream_node)
        for a in template_catalog.actuators
    )
    if event_signature != template_signature:
        raise ValueError(
            "native-controls template actuator identity/topology differs from the event INP"
        )


def build_runtime_inp(
    source: str | Path,
    destination: str | Path,
    *,
    native_controls: bool,
    swmm_threads: int | None = None,
    native_controls_template: str | Path | None = None,
    source_sha256: str | None = None,
) -> RuntimeInpContract:
    """Create a policy-isolated runtime INP without changing event forcing/hydraulics.

    The event ``source`` is always authoritative for rainfall, DWF, initial conditions, dates,
    storage and hydraulic geometry. ``native_controls=False`` removes only executable lines
    inside ``[CONTROLS]``. For ``native_controls=True`` an optional frozen network template may
    supply the native ``[CONTROLS]`` body while every non-control section continues to come from
    the event source. This is the required construction when event INPs intentionally carry the
    forcing/DWF but not the native rule set.

    ``THREADS`` is execution-only. Relative external ``FILE`` inputs are rewritten to absolute
    paths before relocation so the exact forcing bytes remain bound.
    """

    src = Path(source).resolve()
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(f"runtime source INP is missing: {src}")
    if swmm_threads is not None and swmm_threads <= 0:
        raise ValueError("swmm_threads must be positive or None")
    if native_controls_template is not None and not native_controls:
        raise ValueError("native_controls_template is valid only when native_controls=True")

    template: Path | None = None
    template_body: list[str] | None = None
    if native_controls_template is not None:
        template = Path(native_controls_template).resolve()
        if not template.is_file():
            raise ValueError(f"native-controls template is missing: {template}")
        if not section_has_payload(template, "CONTROLS"):
            raise ValueError("native-controls template contains no executable [CONTROLS]")
        _assert_controls_template_compatible(src, template)
        template_body = _section_body_lines(template, "CONTROLS")

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

        if section == "CONTROLS" and (not native_controls or template_body is not None):
            # Keep section structure but remove the source rule body. A frozen template, when
            # supplied, is inserted after all event-source processing is complete.
            if not core.split(";", 1)[0].strip() and template_body is None:
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

    if template_body is not None:
        output = _replace_section_body(
            output, section_name="CONTROLS", replacement=template_body
        )

    if swmm_threads is not None and not threads_seen:
        raise ValueError("[OPTIONS] THREADS was not found; refuse to silently mutate layout")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(output), encoding="utf-8")
    if native_controls:
        assert_native_controls_enabled(dst)
    else:
        assert_native_controls_disabled(dst)
    return RuntimeInpContract(
        source_path=str(src),
        runtime_path=str(dst.resolve()),
        source_sha256=sha256_file(src) if source_sha256 is None else str(source_sha256),
        runtime_sha256=sha256_file(dst),
        native_controls_enabled=bool(native_controls),
        swmm_threads=swmm_threads,
        native_controls_template_path=(None if template is None else str(template)),
        native_controls_template_sha256=(None if template is None else sha256_file(template)),
    )
