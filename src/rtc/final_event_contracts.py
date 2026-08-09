from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .inp_lineage import physical_contract_sha256, scientific_event_contract_sha256
from .tfv_pipeline import sha256_file


FINAL_EVENT_CONTRACT = "FINAL_EVENT_FORCING_LOCK_V1"


def _final_registry(path: str | Path) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"split registry lacks Final event lineage: {missing}")
    registry = registry.copy()
    for column in ("event_id", "rainfall_group", "scientific_split"):
        registry[column] = registry[column].fillna("").astype(str)
    if registry["event_id"].duplicated().any():
        raise ValueError("split registry must contain one authoritative row per event_id")
    final = registry[registry["scientific_split"] == "final"].copy()
    if final.empty:
        raise ValueError("split registry contains no Final events")
    return final


def build_final_event_contracts(
    *,
    split_registry_path: str | Path,
    output_path: str | Path,
    expected_physical_sha256: str,
) -> dict[str, object]:
    """Snapshot the untouched Final event bytes/forcing at Policy-Lock preparation time."""

    final = _final_registry(split_registry_path)
    events: dict[str, dict[str, str]] = {}
    for _, row in final.sort_values("event_id").iterrows():
        event_id = str(row["event_id"])
        inp = Path(str(row["inp_path"])).resolve()
        if not inp.is_file():
            raise ValueError(f"Final event INP is missing: {inp}")
        physical = physical_contract_sha256(inp)
        if physical != str(expected_physical_sha256):
            raise ValueError(
                f"Final event {event_id} uses a different physical network: {physical}"
            )
        events[event_id] = {
            "rainfall_group": str(row["rainfall_group"]),
            "inp_path": str(inp),
            "source_inp_sha256": sha256_file(inp),
            "scientific_event_sha256": scientific_event_contract_sha256(inp),
            "physical_network_sha256": physical,
        }
    payload: dict[str, object] = {
        "contract": FINAL_EVENT_CONTRACT,
        "split_registry_path": str(Path(split_registry_path).resolve()),
        "split_registry_sha256": sha256_file(split_registry_path),
        "physical_network_sha256": str(expected_physical_sha256),
        "event_count": len(events),
        "events": events,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_final_event_contracts(
    path: str | Path,
    *,
    split_registry_path: str | Path,
    expected_physical_sha256: str,
    verify_current_files: bool = True,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != FINAL_EVENT_CONTRACT:
        raise ValueError("invalid Final event forcing lock artifact")
    if str(payload.get("split_registry_sha256", "")) != sha256_file(split_registry_path):
        raise ValueError("Final event forcing lock belongs to a different split registry")
    if str(payload.get("physical_network_sha256", "")) != str(expected_physical_sha256):
        raise ValueError("Final event forcing lock uses a different physical network")
    events = payload.get("events")
    if not isinstance(events, dict) or not events:
        raise ValueError("Final event forcing lock contains no events")

    registry = _final_registry(split_registry_path)
    registry_ids = set(registry["event_id"].astype(str))
    if set(map(str, events)) != registry_ids:
        raise ValueError("Final event forcing lock event set differs from the locked registry")
    registry_group = registry.set_index("event_id")["rainfall_group"].astype(str).to_dict()
    registry_path = registry.set_index("event_id")["inp_path"].astype(str).to_dict()

    for event_id, raw in events.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid Final event lock row: {event_id}")
        if str(raw.get("rainfall_group", "")) != str(registry_group[event_id]):
            raise ValueError(f"Final event {event_id} rainfall_group changed after lock")
        if str(raw.get("physical_network_sha256", "")) != str(expected_physical_sha256):
            raise ValueError(f"Final event {event_id} physical network differs from lock")
        if verify_current_files:
            inp = Path(str(registry_path[event_id])).resolve()
            if not inp.is_file():
                raise ValueError(f"locked Final event INP disappeared: {inp}")
            if str(raw.get("inp_path", "")) != str(inp):
                raise ValueError(f"Final event {event_id} registry INP path changed after lock")
            if str(raw.get("source_inp_sha256", "")) != sha256_file(inp):
                raise ValueError(f"Final event {event_id} source INP bytes changed after lock")
            if str(raw.get("scientific_event_sha256", "")) != scientific_event_contract_sha256(inp):
                raise ValueError(f"Final event {event_id} forcing/external FILE bytes changed after lock")
            if physical_contract_sha256(inp) != str(expected_physical_sha256):
                raise ValueError(f"Final event {event_id} physical network changed after lock")
    return payload
