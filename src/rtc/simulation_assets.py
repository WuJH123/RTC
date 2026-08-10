from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .inp_lineage import canonical_scientific_event_contract, physical_contract_sha256
from .inp_runtime import sha256_file
from .replay_prefix import load_checkpoint_reference

SIMULATION_IDENTITY_CONTRACT = "RTC_SIMULATION_IDENTITY_V1_STATE_ACTION_ENGINE_BOUND"
ASSET_REGISTRY_CONTRACT = "RTC_SIMULATION_ASSET_REGISTRY_V1_LOCAL_ONLY"
D2_EXECUTION_SEMANTICS = "D2_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED"

VALID_REUSABLE = "VALID_REUSABLE"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
INVALID = "INVALID"
STALE = "STALE"
FAILED = "FAILED"
PENDING = "PENDING"
REBUILDABLE_CACHE = "REBUILDABLE_CACHE"
QUALIFICATIONS = {
    VALID_REUSABLE,
    DIAGNOSTIC_ONLY,
    INVALID,
    STALE,
    FAILED,
    PENDING,
    REBUILDABLE_CACHE,
}

# These OPTIONS change only how long an already-identical event is allowed to continue.
# They must not split one physical checkpoint/action family into separate identities.
_TAIL_ONLY_OPTIONS = {"END_DATE", "END_TIME", "REPORT_END_DATE", "REPORT_END_TIME"}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def event_prefix_family_sha256(inp_path: str | Path) -> str:
    """Hash event physics/forcing/start clock while ignoring only recovery-tail END options.

    This intentionally keeps START/REPORT_START, rainfall rows, DWF, physical sections and all
    other scientific content bound. Extending a prepared event from a 480-min to a 600-min
    recovery tail therefore remains in the same family; changing warm-up, rainfall, DWF or the
    physical network does not.
    """

    contract = canonical_scientific_event_contract(inp_path)
    cleaned: dict[str, list[str]] = {}
    for section, rows in contract.items():
        if section != "OPTIONS":
            cleaned[section] = list(rows)
            continue
        kept: list[str] = []
        for row in rows:
            tokens = row.split()
            if tokens and tokens[0].upper() in _TAIL_ONLY_OPTIONS:
                continue
            kept.append(row)
        cleaned[section] = kept
    return sha256_json(cleaned)


def _option_clock(inp_path: str | Path) -> tuple[datetime, datetime]:
    options: dict[str, str] = {}
    section = ""
    for raw in Path(inp_path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            continue
        if section != "OPTIONS":
            continue
        body = raw.split(";", 1)[0].strip().split()
        if len(body) >= 2:
            options[body[0].upper()] = body[1]
    required = ("START_DATE", "START_TIME", "END_DATE", "END_TIME")
    missing = [key for key in required if key not in options]
    if missing:
        raise ValueError(f"SWMM INP lacks time OPTIONS {missing}: {inp_path}")

    def parse_date(value: str):
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass
        raise ValueError(f"unsupported SWMM date token: {value}")

    def parse_time(value: str):
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(value, fmt).time()
            except ValueError:
                pass
        raise ValueError(f"unsupported SWMM time token: {value}")

    start = datetime.combine(parse_date(options["START_DATE"]), parse_time(options["START_TIME"]))
    end = datetime.combine(parse_date(options["END_DATE"]), parse_time(options["END_TIME"]))
    if end <= start:
        raise ValueError(f"SWMM END must be after START: {inp_path}")
    return start, end


def simulation_available_seconds(inp_path: str | Path) -> int:
    start, end = _option_clock(inp_path)
    return int(round((end - start).total_seconds()))


def endpoint_preflight_from_available(
    inp_path: str | Path,
    *,
    available_end_seconds: int,
    checkpoint_seconds: int,
    horizon_seconds: int,
) -> dict[str, int | str]:
    """Build the endpoint evidence after the event OPTIONS clock was parsed once."""

    if checkpoint_seconds <= 0 or horizon_seconds <= 0:
        raise ValueError("checkpoint/horizon seconds must be positive")
    available = int(available_end_seconds)
    required = int(checkpoint_seconds) + int(horizon_seconds)
    if required > available:
        raise ValueError(
            "D2 endpoint preflight failed before SWMM launch: "
            f"checkpoint+horizon={required}s exceeds event available_end={available}s; "
            "prepare a longer recovery tail or choose an earlier checkpoint"
        )
    return {
        "contract": "RTC_EXACT_ENDPOINT_PREFLIGHT_V1",
        "available_end_seconds": available,
        "required_end_seconds": required,
        "remaining_margin_seconds": available - required,
        "inp_path": str(Path(inp_path).resolve()),
    }


def assert_endpoint_available(
    inp_path: str | Path, *, checkpoint_seconds: int, horizon_seconds: int
) -> dict[str, int | str]:
    """Fail before launching SWMM if a requested exact endpoint cannot exist."""

    return endpoint_preflight_from_available(
        inp_path,
        available_end_seconds=simulation_available_seconds(inp_path),
        checkpoint_seconds=checkpoint_seconds,
        horizon_seconds=horizon_seconds,
    )


def checkpoint_state_sha256_from_values(
    *,
    elapsed_seconds: int,
    node_ids: tuple[str, ...],
    state_si: np.ndarray,
    actuator_ids: tuple[str, ...],
    current_setting: np.ndarray,
    swmm_engine_version: str,
) -> str:
    """Hash one exact checkpoint state using the frozen simulation identity contract."""

    h = hashlib.sha256()
    h.update(SIMULATION_IDENTITY_CONTRACT.encode("utf-8"))
    h.update(str(int(elapsed_seconds)).encode("ascii"))
    h.update(str(swmm_engine_version).encode("utf-8"))
    h.update(canonical_json(tuple(node_ids)).encode("utf-8"))
    h.update(np.asarray(state_si, dtype="<f8").tobytes(order="C"))
    h.update(canonical_json(tuple(actuator_ids)).encode("utf-8"))
    h.update(np.asarray(current_setting, dtype="<f8").tobytes(order="C"))
    return h.hexdigest()


def checkpoint_state_sha256(metadata_path: str | Path, *, elapsed_seconds: int) -> str:
    """Hash the exact pre-action hydraulic/readback state rather than its file location."""

    reference = load_checkpoint_reference(metadata_path, elapsed_seconds=elapsed_seconds)
    return checkpoint_state_sha256_from_values(
        elapsed_seconds=reference.elapsed_seconds,
        node_ids=reference.node_ids,
        state_si=reference.state_si,
        actuator_ids=reference.actuator_ids,
        current_setting=reference.current_setting,
        swmm_engine_version=reference.swmm_engine_version,
    )


def d2_identity_from_precomputed(
    *,
    physical_network_sha256: str,
    event_prefix_family_sha256: str,
    checkpoint_seconds: int,
    checkpoint_state_sha256_value: str,
    candidate_action_sha256: str,
    swmm_engine_version: str,
    stride_seconds: int,
    horizon_seconds: int,
) -> tuple[str, str, dict[str, object]]:
    """Return D2 identity from immutable preflight values without rereading source assets."""

    if min(checkpoint_seconds, stride_seconds, horizon_seconds) <= 0:
        raise ValueError("D2 identity timing values must be positive")
    family = {
        "identity_contract": SIMULATION_IDENTITY_CONTRACT,
        "kind": "D2_LOCAL_STEP",
        "execution_semantics": D2_EXECUTION_SEMANTICS,
        "physical_network_sha256": str(physical_network_sha256),
        "event_prefix_family_sha256": str(event_prefix_family_sha256),
        "checkpoint_elapsed_seconds": int(checkpoint_seconds),
        "checkpoint_state_sha256": str(checkpoint_state_sha256_value),
        "candidate_action_sha256": str(candidate_action_sha256),
        "native_controls_enabled": False,
        "swmm_engine_version": str(swmm_engine_version),
        "record_stride_seconds": int(stride_seconds),
    }
    family_key = sha256_json(family)
    payload = {**family, "horizon_seconds": int(horizon_seconds)}
    simulation_key = sha256_json(payload)
    return simulation_key, family_key, payload


def d2_identity(
    *,
    inp_path: str | Path,
    reference_metadata_path: str | Path,
    checkpoint_seconds: int,
    candidate_action_sha256: str,
    swmm_engine_version: str,
    stride_seconds: int,
    horizon_seconds: int,
) -> tuple[str, str, dict[str, object]]:
    """Return ``(simulation_key, family_key, payload)`` for one D2 physical experiment."""

    if min(checkpoint_seconds, stride_seconds, horizon_seconds) <= 0:
        raise ValueError("D2 identity timing values must be positive")
    return d2_identity_from_precomputed(
        physical_network_sha256=physical_contract_sha256(inp_path),
        event_prefix_family_sha256=event_prefix_family_sha256(inp_path),
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_state_sha256_value=checkpoint_state_sha256(
            reference_metadata_path, elapsed_seconds=int(checkpoint_seconds)
        ),
        candidate_action_sha256=candidate_action_sha256,
        swmm_engine_version=swmm_engine_version,
        stride_seconds=stride_seconds,
        horizon_seconds=horizon_seconds,
    )


def artifact_hashes_from_metadata(metadata_path: str | Path) -> dict[str, str]:
    path = Path(metadata_path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    fields: list[str] = ["compact_file", "node_statistics_file"]
    snapshot = meta.get("horizon_snapshot_files")
    if isinstance(snapshot, dict):
        fields.extend(str(value) for value in snapshot.values())
    hashes: dict[str, str] = {}
    for field in fields[:2]:
        raw = meta.get(field)
        if not raw:
            continue
        artifact = path.parent / str(raw)
        if not artifact.is_file():
            raise ValueError(f"metadata artifact is missing: {artifact}")
        hashes[str(artifact.resolve())] = sha256_file(artifact)
    if isinstance(snapshot, dict):
        for raw in snapshot.values():
            artifact = path.parent / str(raw)
            if not artifact.is_file():
                raise ValueError(f"snapshot artifact is missing: {artifact}")
            hashes[str(artifact.resolve())] = sha256_file(artifact)
    return dict(sorted(hashes.items()))


@dataclass(frozen=True)
class AssetHit:
    simulation_key: str
    family_key: str
    kind: str
    horizon_seconds: int
    metadata_path: str
    qualification: str
    exact: bool


class RegistrySnapshot:
    """Read-only in-memory registry view for a single preflight pass.

    SQLite is still the authority.  The snapshot only removes repeated connection/query setup;
    metadata and artifact hashes are verified lazily on the first actual hit.
    """

    def __init__(self, registry: "SimulationAssetRegistry", rows: Iterable[Mapping[str, object]]):
        self._registry = registry
        self._exact: dict[str, Mapping[str, object]] = {}
        self._families: dict[str, list[Mapping[str, object]]] = {}
        self._verified: dict[tuple[str, bool], AssetHit | None] = {}
        for row in rows:
            simulation_key = str(row["simulation_key"])
            family_key = str(row["family_key"])
            self._exact[simulation_key] = row
            self._families.setdefault(family_key, []).append(row)
        for family_rows in self._families.values():
            family_rows.sort(key=lambda row: int(row["horizon_seconds"]))

    def _verify(self, row: Mapping[str, object], *, exact: bool) -> AssetHit | None:
        key = (str(row["simulation_key"]), bool(exact))
        if key not in self._verified:
            self._verified[key] = self._registry._verified_hit(row, exact=exact)
        return self._verified[key]

    def lookup_exact(self, simulation_key: str) -> AssetHit | None:
        row = self._exact.get(str(simulation_key))
        return None if row is None else self._verify(row, exact=True)

    def lookup_covering(self, family_key: str, *, horizon_seconds: int) -> AssetHit | None:
        for row in self._families.get(str(family_key), ()):
            if int(row["horizon_seconds"]) < int(horizon_seconds):
                continue
            hit = self._verify(row, exact=False)
            if hit is not None:
                return hit
        return None


class SimulationAssetRegistry:
    """SQLite-backed local registry; large hydraulic arrays remain on the user's local disk."""

    def __init__(self, asset_root: str | Path):
        self.root = Path(asset_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_dir = self.root / "_registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.registry_dir / "simulations.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulations (
                    simulation_key TEXT PRIMARY KEY,
                    family_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    horizon_seconds INTEGER NOT NULL,
                    metadata_path TEXT NOT NULL,
                    metadata_sha256 TEXT NOT NULL,
                    qualification TEXT NOT NULL,
                    qualification_reason TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    artifact_hashes_json TEXT NOT NULL,
                    registered_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sim_family_horizon "
                "ON simulations(family_key, horizon_seconds)"
            )

    def register(
        self,
        *,
        simulation_key: str,
        family_key: str,
        kind: str,
        horizon_seconds: int,
        metadata_path: str | Path,
        identity: Mapping[str, object],
        qualification: str = VALID_REUSABLE,
        qualification_reason: str = "verified generated hydraulic asset",
    ) -> None:
        self.register_many(
            [
                {
                    "simulation_key": simulation_key,
                    "family_key": family_key,
                    "kind": kind,
                    "horizon_seconds": horizon_seconds,
                    "metadata_path": metadata_path,
                    "identity": identity,
                    "qualification": qualification,
                    "qualification_reason": qualification_reason,
                }
            ]
        )

    def register_many(self, records: Iterable[Mapping[str, object]]) -> None:
        """Register a batch with one connection/transaction."""

        prepared: list[tuple[object, ...]] = []
        for record in records:
            qualification = str(record.get("qualification", VALID_REUSABLE))
            if qualification not in QUALIFICATIONS:
                raise ValueError(f"unknown asset qualification: {qualification}")
            path = Path(str(record["metadata_path"])).resolve()
            if not path.is_file():
                raise ValueError(f"cannot register missing metadata: {path}")
            identity = record.get("identity")
            if not isinstance(identity, Mapping):
                raise ValueError(f"asset identity must be a mapping: {path}")
            artifact_hashes = artifact_hashes_from_metadata(path)
            prepared.append(
                (
                    str(record["simulation_key"]),
                    str(record["family_key"]),
                    str(record["kind"]),
                    int(record["horizon_seconds"]),
                    str(path),
                    sha256_file(path),
                    qualification,
                    str(record.get("qualification_reason", "verified generated hydraulic asset")),
                    canonical_json(dict(identity)),
                    canonical_json(artifact_hashes),
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                )
            )
        if not prepared:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO simulations(
                    simulation_key,family_key,kind,horizon_seconds,metadata_path,
                    metadata_sha256,qualification,qualification_reason,identity_json,
                    artifact_hashes_json,registered_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(simulation_key) DO UPDATE SET
                    family_key=excluded.family_key,
                    kind=excluded.kind,
                    horizon_seconds=excluded.horizon_seconds,
                    metadata_path=excluded.metadata_path,
                    metadata_sha256=excluded.metadata_sha256,
                    qualification=excluded.qualification,
                    qualification_reason=excluded.qualification_reason,
                    identity_json=excluded.identity_json,
                    artifact_hashes_json=excluded.artifact_hashes_json,
                    registered_utc=excluded.registered_utc
                """,
                prepared,
            )

    def preflight_snapshot(self) -> RegistrySnapshot:
        """Load all registry rows once; actual artifact verification remains lazy."""

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM simulations").fetchall()
        return RegistrySnapshot(self, [dict(row) for row in rows])

    def _verified_hit(self, row: sqlite3.Row, *, exact: bool) -> AssetHit | None:
        if str(row["qualification"]) != VALID_REUSABLE:
            return None
        metadata = Path(str(row["metadata_path"]))
        if not metadata.is_file() or sha256_file(metadata) != str(row["metadata_sha256"]):
            return None
        try:
            artifacts = json.loads(str(row["artifact_hashes_json"]))
        except json.JSONDecodeError:
            return None
        if not isinstance(artifacts, dict):
            return None
        for raw, expected in artifacts.items():
            artifact = Path(str(raw))
            if not artifact.is_file() or sha256_file(artifact) != str(expected):
                return None
        return AssetHit(
            simulation_key=str(row["simulation_key"]),
            family_key=str(row["family_key"]),
            kind=str(row["kind"]),
            horizon_seconds=int(row["horizon_seconds"]),
            metadata_path=str(metadata),
            qualification=str(row["qualification"]),
            exact=exact,
        )

    def lookup_exact(self, simulation_key: str) -> AssetHit | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM simulations WHERE simulation_key=?", (str(simulation_key),)
            ).fetchone()
        return None if row is None else self._verified_hit(row, exact=True)

    def lookup_covering(self, family_key: str, *, horizon_seconds: int) -> AssetHit | None:
        """Find a longer trajectory in the same family.

        A covering hit is valid for trajectory/timing prefix views. It is *not* by itself proof
        of exact shorter-horizon cumulative SWMM statistics unless the metadata contains a
        matching ``horizon_snapshot_files`` endpoint.
        """

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM simulations WHERE family_key=? AND horizon_seconds>=? "
                "ORDER BY horizon_seconds ASC",
                (str(family_key), int(horizon_seconds)),
            ).fetchall()
        for row in rows:
            hit = self._verified_hit(row, exact=False)
            if hit is not None:
                return hit
        return None

    def rows(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM simulations ORDER BY kind,family_key,horizon_seconds"
            ).fetchall()
        return [dict(row) for row in rows]

    def audit(self) -> dict[str, object]:
        rows = self.rows()
        counts: dict[str, int] = {}
        missing_or_changed = 0
        for row in rows:
            qualification = str(row["qualification"])
            counts[qualification] = counts.get(qualification, 0) + 1
            metadata = Path(str(row["metadata_path"]))
            if not metadata.is_file() or sha256_file(metadata) != str(row["metadata_sha256"]):
                missing_or_changed += 1
        return {
            "contract": ASSET_REGISTRY_CONTRACT,
            "asset_root": str(self.root),
            "registry": str(self.db_path),
            "assets": len(rows),
            "qualification_counts": counts,
            "metadata_missing_or_changed": missing_or_changed,
        }


def _stamped_identity_record(
    metadata_path: str | Path,
    *,
    expected_data_contract: str,
    expected_kind: str,
    qualification: str,
    qualification_reason: str,
) -> dict[str, object]:
    """Validate an identity stamped by the branch runner without recomputing its inputs."""

    path = Path(metadata_path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("data_contract") != expected_data_contract:
        raise ValueError(f"unsupported stamped metadata contract: {path}")
    verification = meta.get("same_prefix_verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"metadata lacks passed exact-prefix verification: {path}")
    identity = meta.get("simulation_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"metadata lacks stamped simulation identity: {path}")
    if meta.get("simulation_identity_contract") != SIMULATION_IDENTITY_CONTRACT:
        raise ValueError(f"metadata lacks the frozen simulation identity contract: {path}")
    if identity.get("identity_contract") != SIMULATION_IDENTITY_CONTRACT:
        raise ValueError(f"metadata has incompatible simulation identity contract: {path}")
    if identity.get("kind") != expected_kind:
        raise ValueError(f"metadata has incompatible simulation identity kind: {path}")
    simulation_key = str(meta.get("simulation_identity_sha256", ""))
    family_key = str(meta.get("simulation_family_sha256", ""))
    if not simulation_key or not family_key:
        raise ValueError(f"metadata lacks stamped simulation identity hashes: {path}")
    if sha256_json(identity) != simulation_key:
        raise ValueError(f"stamped simulation identity hash mismatch: {path}")
    family_payload = {key: value for key, value in identity.items() if key != "horizon_seconds"}
    if sha256_json(family_payload) != family_key:
        raise ValueError(f"stamped simulation family hash mismatch: {path}")
    horizon_seconds = int(identity.get("horizon_seconds", 0))
    if horizon_seconds <= 0:
        raise ValueError(f"stamped simulation identity lacks a positive horizon: {path}")
    return {
        "simulation_key": simulation_key,
        "family_key": family_key,
        "kind": expected_kind,
        "horizon_seconds": horizon_seconds,
        "metadata_path": path,
        "identity": identity,
        "qualification": qualification,
        "qualification_reason": qualification_reason,
    }


def register_stamped_d2_metadata(
    registry: SimulationAssetRegistry,
    metadata_path: str | Path,
    *,
    qualification: str = VALID_REUSABLE,
    qualification_reason: str = "exact-prefix D2 branch verified",
) -> tuple[str, str]:
    record = _stamped_identity_record(
        metadata_path,
        expected_data_contract=D2_EXECUTION_SEMANTICS,
        expected_kind="D2_LOCAL_STEP",
        qualification=qualification,
        qualification_reason=qualification_reason,
    )
    registry.register_many([record])
    return str(record["simulation_key"]), str(record["family_key"])


def register_stamped_d2_metadata_many(
    registry: SimulationAssetRegistry,
    metadata_paths: Iterable[str | Path],
    *,
    qualification: str = VALID_REUSABLE,
    qualification_reason: str = "exact-prefix D2 branch verified",
) -> list[tuple[str, str]]:
    records = [
        _stamped_identity_record(
            path,
            expected_data_contract=D2_EXECUTION_SEMANTICS,
            expected_kind="D2_LOCAL_STEP",
            qualification=qualification,
            qualification_reason=qualification_reason,
        )
        for path in metadata_paths
    ]
    registry.register_many(records)
    return [(str(record["simulation_key"]), str(record["family_key"])) for record in records]


def register_d2_metadata(
    registry: SimulationAssetRegistry,
    metadata_path: str | Path,
    *,
    qualification: str = VALID_REUSABLE,
    qualification_reason: str = "exact-prefix D2 branch verified",
) -> tuple[str, str]:
    path = Path(metadata_path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("data_contract") != D2_EXECUTION_SEMANTICS:
        raise ValueError(f"not a supported D2 metadata contract: {path}")
    if (
        isinstance(meta.get("simulation_identity"), dict)
        and meta.get("simulation_identity_contract")
        and meta.get("simulation_identity_sha256")
        and meta.get("simulation_family_sha256")
    ):
        return register_stamped_d2_metadata(
            registry,
            path,
            qualification=qualification,
            qualification_reason=qualification_reason,
        )
    verification = meta.get("same_prefix_verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise ValueError(f"D2 metadata lacks passed exact-prefix verification: {path}")
    reference = str(verification.get("reference_metadata_path", ""))
    if not reference:
        raise ValueError(f"D2 metadata lacks reference metadata path: {path}")
    checkpoint_seconds = int(meta["checkpoint_minutes"]) * 60
    horizon_seconds = int(meta["horizon_minutes"]) * 60
    simulation_key, family_key, identity = d2_identity(
        inp_path=str(meta["inp_path"]),
        reference_metadata_path=reference,
        checkpoint_seconds=checkpoint_seconds,
        candidate_action_sha256=str(meta["candidate_action_sha256"]),
        swmm_engine_version=str(meta["swmm_engine_version"]),
        stride_seconds=int(meta["python_intervention_seconds"]),
        horizon_seconds=horizon_seconds,
    )
    registry.register(
        simulation_key=simulation_key,
        family_key=family_key,
        kind="D2_LOCAL_STEP",
        horizon_seconds=horizon_seconds,
        metadata_path=path,
        identity=identity,
        qualification=qualification,
        qualification_reason=qualification_reason,
    )
    return simulation_key, family_key


def index_d2_metadata_paths(
    asset_root: str | Path,
    metadata_paths: Iterable[str | Path],
    *,
    qualification: str = VALID_REUSABLE,
    qualification_reason: str = "existing exact-prefix D2 evidence indexed after audit",
) -> dict[str, object]:
    registry = SimulationAssetRegistry(asset_root)
    indexed = 0
    failures: list[dict[str, str]] = []
    for raw in metadata_paths:
        try:
            register_d2_metadata(
                registry,
                raw,
                qualification=qualification,
                qualification_reason=qualification_reason,
            )
            indexed += 1
        except Exception as exc:
            failures.append({"metadata_path": str(raw), "error": str(exc)})
    return {
        "contract": "RTC_EXISTING_D2_ASSET_INDEX_V1",
        "asset_root": str(Path(asset_root).resolve()),
        "indexed": indexed,
        "failed": len(failures),
        "failures": failures,
        "audit": registry.audit(),
    }
