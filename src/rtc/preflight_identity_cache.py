from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .generation_contract import atomic_json_write
from .inp_lineage import (
    external_event_file_hashes,
    physical_contract_sha256,
)
from .inp_runtime import sha256_file
from .simulation_assets import (
    checkpoint_state_sha256_from_values,
    endpoint_preflight_from_available,
    event_prefix_family_sha256,
    simulation_available_seconds,
)


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def capture(cls, path: str | Path) -> FileIdentity:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"preflight identity file is missing: {resolved}")
        stat = resolved.stat()
        return cls(
            path=str(resolved),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            sha256=sha256_file(resolved),
        )

    def assert_unchanged(self) -> None:
        path = Path(self.path)
        if not path.is_file():
            raise RuntimeError(f"preflight identity input disappeared: {path}")
        stat = path.stat()
        if int(stat.st_size) != self.size or int(stat.st_mtime_ns) != self.mtime_ns:
            raise RuntimeError(
                "preflight identity input changed after cache capture: "
                f"{path} (mtime/size changed)"
            )
        current = sha256_file(path)
        if current != self.sha256:
            raise RuntimeError(f"preflight identity input changed after cache capture: {path}")


@dataclass(frozen=True)
class EventIdentityContext:
    source_path: str
    source_file: FileIdentity
    physical_network_sha256: str
    event_prefix_family_sha256: str
    available_end_seconds: int
    external_event_file_sha256: dict[str, str]

    def endpoint_preflight(self, *, checkpoint_seconds: int, horizon_seconds: int) -> dict[str, int | str]:
        return endpoint_preflight_from_available(
            self.source_path,
            available_end_seconds=self.available_end_seconds,
            checkpoint_seconds=checkpoint_seconds,
            horizon_seconds=horizon_seconds,
        )

    def assert_unchanged(self) -> None:
        self.source_file.assert_unchanged()
        for raw, expected in self.external_event_file_sha256.items():
            path = Path(raw)
            if not path.is_file() or sha256_file(path) != str(expected):
                raise RuntimeError(f"external event forcing changed after cache capture: {path}")


@dataclass(frozen=True)
class ReferenceTrajectoryContext:
    metadata_path: str
    metadata_file: FileIdentity
    compact_file: FileIdentity
    swmm_engine_version: str
    checkpoint_state_sha256_by_elapsed: dict[int, str]

    @property
    def lineage(self) -> dict[str, str]:
        return {
            "reference_metadata_path": self.metadata_path,
            "reference_metadata_sha256": self.metadata_file.sha256,
            "reference_compact_path": self.compact_file.path,
            "reference_compact_sha256": self.compact_file.sha256,
            "reference_swmm_engine_version": self.swmm_engine_version,
        }

    def checkpoint_state_sha256(self, elapsed_seconds: int) -> str:
        try:
            return self.checkpoint_state_sha256_by_elapsed[int(elapsed_seconds)]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint state was not loaded for {elapsed_seconds}s: {self.metadata_path}"
            ) from exc

    def assert_unchanged(self) -> None:
        self.metadata_file.assert_unchanged()
        self.compact_file.assert_unchanged()


class PreflightProgress:
    """Atomic, human-readable progress ledger for a census-only or execution preflight."""

    def __init__(self, path: str | Path, *, total: int):
        self.path = Path(path)
        self.started = time.monotonic()
        self.payload: dict[str, object] = {
            "contract": "RTC_PREFLIGHT_PROGRESS_V1",
            "processed": 0,
            "total": int(total),
            "stage": "INITIALIZING",
            "events_cached": 0,
            "references_cached": 0,
            "checkpoints_cached": 0,
            "endpoint_invalid": 0,
            "exact_asset_candidates": 0,
            "covering_asset_candidates": 0,
            "elapsed_seconds": 0.0,
        }
        self.write()

    def update(self, **values: object) -> None:
        self.payload.update(values)
        self.payload["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        self.write()

    def write(self) -> None:
        atomic_json_write(self.path, self.payload)


class PreflightIdentityCache:
    """Shared immutable contexts for D2/D3 identity and endpoint preflight."""

    CACHE_CONTRACT = "RTC_PREFLIGHT_IDENTITY_CACHE_V1"

    def __init__(
        self,
        *,
        events: Mapping[str, EventIdentityContext],
        references: Mapping[str, ReferenceTrajectoryContext],
    ):
        self.events = dict(events)
        self.references = dict(references)

    @staticmethod
    def _normalise_requests(
        requests: Mapping[str | Path, Iterable[int]],
    ) -> dict[str, tuple[int, ...]]:
        return {
            str(Path(path).expanduser().resolve()): tuple(sorted({int(value) for value in values}))
            for path, values in requests.items()
        }

    @classmethod
    def build(
        cls,
        *,
        event_paths_to_checkpoints: Mapping[str | Path, Iterable[int]],
        reference_paths_to_checkpoints: Mapping[str | Path, Iterable[int]],
        cache_path: str | Path | None = None,
        progress: PreflightProgress | None = None,
    ) -> PreflightIdentityCache:
        event_requests = cls._normalise_requests(event_paths_to_checkpoints)
        reference_requests = cls._normalise_requests(reference_paths_to_checkpoints)
        if cache_path is not None and Path(cache_path).is_file():
            cached = cls._load(cache_path)
            cached._assert_request_coverage(event_requests, reference_requests)
            cached.assert_unchanged()
            if progress is not None:
                progress.update(
                    stage="CACHE_REUSED",
                    events_cached=len(cached.events),
                    references_cached=len(cached.references),
                    checkpoints_cached=sum(
                        len(context.checkpoint_state_sha256_by_elapsed)
                        for context in cached.references.values()
                    ),
                )
            return cached

        events: dict[str, EventIdentityContext] = {}
        if progress is not None:
            progress.update(stage="EVENT_CONTEXT")
        for source in event_requests:
            source_path = Path(source)
            events[source] = EventIdentityContext(
                source_path=source,
                source_file=FileIdentity.capture(source_path),
                physical_network_sha256=physical_contract_sha256(source_path),
                event_prefix_family_sha256=event_prefix_family_sha256(source_path),
                available_end_seconds=simulation_available_seconds(source_path),
                external_event_file_sha256=external_event_file_hashes(source_path),
            )
            if progress is not None:
                progress.update(stage="EVENT_CONTEXT", events_cached=len(events))

        references: dict[str, ReferenceTrajectoryContext] = {}
        if progress is not None:
            progress.update(stage="REFERENCE_CONTEXT", references_cached=0)
        for metadata, checkpoints in reference_requests.items():
            metadata_path = Path(metadata)
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise TypeError(
                    "checkpoint reference metadata must be a JSON object: "
                    f"{metadata_path}"
                )
            compact_name = meta.get("compact_file")
            if not compact_name:
                raise ValueError(f"checkpoint reference requires compact trajectory evidence: {metadata_path}")
            compact_path = metadata_path.parent / str(compact_name)
            if not compact_path.is_file():
                raise ValueError(f"checkpoint reference compact file is missing: {compact_path}")
            engine = str(meta.get("swmm_engine_version", "")).strip()
            if not engine:
                raise ValueError(f"checkpoint reference lacks SWMM engine version: {metadata_path}")
            state_hashes: dict[int, str] = {}
            with np.load(compact_path, allow_pickle=False) as raw:
                elapsed = raw["elapsed_seconds"].astype(np.int64)
                node_ids = tuple(raw["node_ids"].astype(str).tolist())
                actuator_ids = tuple(raw["actuator_ids"].astype(str).tolist())
                state_values = raw["state_si"]
                current_values = raw["current_setting"]
                for seconds in checkpoints:
                    matches = np.flatnonzero(elapsed == int(seconds))
                    if matches.size != 1:
                        raise ValueError(
                            "checkpoint reference requires exactly one sample at "
                            f"{seconds}s: {compact_path}"
                        )
                    index = int(matches[0])
                    state_hashes[int(seconds)] = checkpoint_state_sha256_from_values(
                        elapsed_seconds=int(seconds),
                        node_ids=node_ids,
                        state_si=state_values[index].astype(np.float64),
                        actuator_ids=actuator_ids,
                        current_setting=current_values[index].astype(np.float64),
                        swmm_engine_version=engine,
                    )
            references[metadata] = ReferenceTrajectoryContext(
                metadata_path=metadata,
                metadata_file=FileIdentity.capture(metadata_path),
                compact_file=FileIdentity.capture(compact_path),
                swmm_engine_version=engine,
                checkpoint_state_sha256_by_elapsed=state_hashes,
            )
            if progress is not None:
                progress.update(
                    stage="REFERENCE_CONTEXT",
                    references_cached=len(references),
                    checkpoints_cached=sum(
                        len(context.checkpoint_state_sha256_by_elapsed)
                        for context in references.values()
                    ),
                )

        cache = cls(events=events, references=references)
        if cache_path is not None:
            cache.save(cache_path)
        return cache

    def event(self, path: str | Path) -> EventIdentityContext:
        key = str(Path(path).expanduser().resolve())
        try:
            return self.events[key]
        except KeyError as exc:
            raise KeyError(f"event identity context was not prepared: {key}") from exc

    def reference(self, path: str | Path) -> ReferenceTrajectoryContext:
        key = str(Path(path).expanduser().resolve())
        try:
            return self.references[key]
        except KeyError as exc:
            raise KeyError(f"reference identity context was not prepared: {key}") from exc

    def assert_unchanged(self) -> None:
        for context in self.events.values():
            context.assert_unchanged()
        for context in self.references.values():
            context.assert_unchanged()

    def _assert_request_coverage(
        self,
        event_requests: Mapping[str, tuple[int, ...]],
        reference_requests: Mapping[str, tuple[int, ...]],
    ) -> None:
        if set(event_requests) != set(self.events) or set(reference_requests) != set(self.references):
            raise RuntimeError("preflight identity cache does not match the current manifest assets")
        for path, requested in reference_requests.items():
            available = self.references[path].checkpoint_state_sha256_by_elapsed
            if not set(requested).issubset(available):
                raise RuntimeError(
                    f"preflight identity cache lacks checkpoint states for {path}: {requested}"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "contract": self.CACHE_CONTRACT,
            "events": [asdict(context) for context in self.events.values()],
            "references": [asdict(context) for context in self.references.values()],
        }

    def save(self, path: str | Path) -> None:
        atomic_json_write(path, self.to_payload())

    @classmethod
    def _load(cls, path: str | Path) -> PreflightIdentityCache:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("contract") != cls.CACHE_CONTRACT:
            raise RuntimeError(f"unsupported preflight identity cache: {path}")
        events: dict[str, EventIdentityContext] = {}
        for item in raw.get("events", []):
            source_file = FileIdentity(**item["source_file"])
            context = EventIdentityContext(
                source_path=str(item["source_path"]),
                source_file=source_file,
                physical_network_sha256=str(item["physical_network_sha256"]),
                event_prefix_family_sha256=str(item["event_prefix_family_sha256"]),
                available_end_seconds=int(item["available_end_seconds"]),
                external_event_file_sha256={
                    str(key): str(value)
                    for key, value in item.get("external_event_file_sha256", {}).items()
                },
            )
            events[context.source_path] = context
        references: dict[str, ReferenceTrajectoryContext] = {}
        for item in raw.get("references", []):
            context = ReferenceTrajectoryContext(
                metadata_path=str(item["metadata_path"]),
                metadata_file=FileIdentity(**item["metadata_file"]),
                compact_file=FileIdentity(**item["compact_file"]),
                swmm_engine_version=str(item["swmm_engine_version"]),
                checkpoint_state_sha256_by_elapsed={
                    int(key): str(value)
                    for key, value in item.get("checkpoint_state_sha256_by_elapsed", {}).items()
                },
            )
            references[context.metadata_path] = context
        return cls(events=events, references=references)
