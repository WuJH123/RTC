from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .event_preparation import EVENT_PREPARATION_CONTRACT
from .inp import discover_actuators, discover_nodes
from .inp_runtime import section_has_payload, sha256_file


READINESS_CONTRACT = "WUHAN_RTC_PRETRAINING_READINESS_V1"
SENSOR_PROVENANCE_CONTRACT = "SENSOR_LAYOUT_PROVENANCE_V1"
RAINFALL_PROVENANCE_CONTRACT = "RAINFALL_PROVENANCE_V1"
ACTUATOR_SCOPE_CONTRACT = "ACTUATOR_SCOPE_V1"


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _reject_placeholders(value: object, *, context: str) -> None:
    if isinstance(value, str):
        upper = value.upper()
        if "REPLACE_" in upper or upper in {"TBD", "UNKNOWN", "PLACEHOLDER"}:
            raise ValueError(f"{context} contains unresolved placeholder: {value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _reject_placeholders(child, context=f"{context}[{i}]")


def _sensor_lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def validate_pretraining_readiness(
    *,
    event_registry_path: str | Path,
    frozen_inp_path: str | Path,
    sensor_layout_path: str | Path,
    sensor_provenance_path: str | Path,
    rainfall_provenance_path: str | Path,
    actuator_scope_path: str | Path,
    history_span_minutes: int,
    minimum_post_rain_tail_minutes: int,
) -> dict[str, object]:
    if history_span_minutes <= 0 or minimum_post_rain_tail_minutes <= 0:
        raise ValueError("history span and minimum post-rain tail must be positive")
    events_path = Path(event_registry_path)
    frozen = Path(frozen_inp_path)
    sensors_path = Path(sensor_layout_path)
    for name, path in {
        "event_registry": events_path,
        "frozen_inp": frozen,
        "sensor_layout": sensors_path,
        "sensor_provenance": Path(sensor_provenance_path),
        "rainfall_provenance": Path(rainfall_provenance_path),
        "actuator_scope": Path(actuator_scope_path),
    }.items():
        if not path.is_file():
            raise ValueError(f"readiness input is missing: {name}: {path}")

    events = pd.read_csv(events_path)
    required = {
        "event_id",
        "rainfall_group",
        "inp_path",
        "scientific_split",
        "event_preparation_contract",
        "prepared_inp_sha256",
        "pre_rain_warmup_minutes",
        "post_rain_tail_minutes",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"prepared event registry lacks readiness fields: {missing}")
    if events.empty or events["event_id"].astype(str).duplicated().any():
        raise ValueError("prepared event registry must contain unique non-empty events")
    if (events["event_preparation_contract"].astype(str) != EVENT_PREPARATION_CONTRACT).any():
        raise ValueError("all events must use the current dry-prefix/recovery-tail preparation contract")
    if (events["pre_rain_warmup_minutes"].astype(float) + 1e-9 < history_span_minutes).any():
        bad = events.loc[
            events["pre_rain_warmup_minutes"].astype(float) + 1e-9 < history_span_minutes,
            "event_id",
        ].astype(str).tolist()
        raise ValueError(f"events lack a full causal pre-rain history: {bad[:10]}")
    if (events["post_rain_tail_minutes"].astype(float) + 1e-9 < minimum_post_rain_tail_minutes).any():
        bad = events.loc[
            events["post_rain_tail_minutes"].astype(float) + 1e-9 < minimum_post_rain_tail_minutes,
            "event_id",
        ].astype(str).tolist()
        raise ValueError(f"events lack the required post-rain recovery tail: {bad[:10]}")
    for _, row in events.iterrows():
        event = Path(str(row["inp_path"]))
        if not event.is_file():
            raise ValueError(f"prepared event INP is missing: {event}")
        if sha256_file(event) != str(row["prepared_inp_sha256"]):
            raise ValueError(f"prepared event INP changed after registry creation: {event}")

    sensor_prov = _json(sensor_provenance_path)
    if sensor_prov.get("contract") != SENSOR_PROVENANCE_CONTRACT:
        raise ValueError("sensor provenance must use SENSOR_LAYOUT_PROVENANCE_V1")
    _reject_placeholders(sensor_prov, context="sensor_provenance")
    if str(sensor_prov.get("sensor_layout_sha256", "")) != sha256_file(sensors_path):
        raise ValueError("sensor provenance hash differs from the active sensor layout")
    if sensor_prov.get("hydraulic_outcomes_used_for_selection") is not False:
        raise ValueError("sensor selection must explicitly state that hydraulic outcomes were not used")
    sensors = _sensor_lines(sensors_path)
    frozen_nodes = tuple(discover_nodes(frozen))
    if not sensors or not set(sensors).issubset(frozen_nodes):
        raise ValueError("sensor layout contains nodes outside the frozen network")

    rainfall = _json(rainfall_provenance_path)
    if rainfall.get("contract") != RAINFALL_PROVENANCE_CONTRACT:
        raise ValueError("rainfall provenance must use RAINFALL_PROVENANCE_V1")
    _reject_placeholders(rainfall, context="rainfall_provenance")
    for field in (
        "source_kind",
        "official_standard_claim",
        "spatial_mode",
        "return_period_scope_years",
        "duration_scope_minutes",
        "pattern_scope",
    ):
        if field not in rainfall:
            raise ValueError(f"rainfall provenance lacks {field}")
    if not isinstance(rainfall["official_standard_claim"], bool):
        raise ValueError("rainfall official_standard_claim must be boolean")
    if str(rainfall["spatial_mode"]).strip() == "":
        raise ValueError("rainfall spatial_mode cannot be empty")

    actuator_scope = _json(actuator_scope_path)
    if actuator_scope.get("contract") != ACTUATOR_SCOPE_CONTRACT:
        raise ValueError("actuator scope must use ACTUATOR_SCOPE_V1")
    _reject_placeholders(actuator_scope, context="actuator_scope")
    mode = str(actuator_scope.get("actuation_scope", ""))
    if mode not in {
        "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY",
        "FIELD_ENGINEERING_VALIDATED",
    }:
        raise ValueError("unsupported actuation_scope")
    field_claim = bool(actuator_scope.get("field_deployment_claim", False))
    if mode == "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY" and field_claim:
        raise ValueError("simulation-only actuator scope cannot make a field-deployment claim")
    catalog = discover_actuators(frozen)
    if int(actuator_scope.get("actuator_count", -1)) != len(catalog.ids):
        raise ValueError("actuator scope count differs from frozen INP")
    if mode == "FIELD_ENGINEERING_VALIDATED":
        capability = actuator_scope.get("capability_map_sha256")
        if not isinstance(capability, str) or len(capability) != 64:
            raise ValueError("field-validated actuation requires a hashed engineering capability map")

    # The frozen network is the only valid native-controls template. Event sources are allowed to
    # carry no controls because Internal-RTC receives this template at runtime; the template itself
    # must contain an executable rule set.
    if not section_has_payload(frozen, "CONTROLS"):
        raise ValueError("frozen native-controls template contains no executable [CONTROLS]")

    return {
        "contract": READINESS_CONTRACT,
        "passed": True,
        "event_registry": str(events_path.resolve()),
        "event_registry_sha256": sha256_file(events_path),
        "events": int(len(events)),
        "rainfall_groups": int(events["rainfall_group"].astype(str).nunique()),
        "history_span_minutes": int(history_span_minutes),
        "minimum_pre_rain_warmup_minutes": float(events["pre_rain_warmup_minutes"].astype(float).min()),
        "minimum_post_rain_tail_minutes": float(events["post_rain_tail_minutes"].astype(float).min()),
        "frozen_inp": str(frozen.resolve()),
        "frozen_inp_sha256": sha256_file(frozen),
        "native_controls_template_verified": True,
        "sensor_layout": str(sensors_path.resolve()),
        "sensor_layout_sha256": sha256_file(sensors_path),
        "sensor_count": len(sensors),
        "sensor_provenance_path": str(Path(sensor_provenance_path).resolve()),
        "sensor_provenance_sha256": sha256_file(sensor_provenance_path),
        "rainfall_provenance_path": str(Path(rainfall_provenance_path).resolve()),
        "rainfall_provenance_sha256": sha256_file(rainfall_provenance_path),
        "rainfall_official_standard_claim": bool(rainfall["official_standard_claim"]),
        "rainfall_spatial_mode": str(rainfall["spatial_mode"]),
        "actuator_scope_path": str(Path(actuator_scope_path).resolve()),
        "actuator_scope_sha256": sha256_file(actuator_scope_path),
        "actuation_scope": mode,
        "field_deployment_claim": field_claim,
        "actuator_count": len(catalog.ids),
        "scientific_claim_scope": (
            "MODEL-BASED SWMM RTC STUDY; field actuation is not certified unless the actuator scope is FIELD_ENGINEERING_VALIDATED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed before production training if event/history/provenance/actuation readiness is unresolved"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--frozen-inp", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--sensor-provenance", required=True)
    parser.add_argument("--rainfall-provenance", required=True)
    parser.add_argument("--actuator-scope", required=True)
    parser.add_argument("--history-span-minutes", type=int, required=True)
    parser.add_argument("--minimum-post-rain-tail-minutes", type=int, default=360)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = validate_pretraining_readiness(
        event_registry_path=args.events,
        frozen_inp_path=args.frozen_inp,
        sensor_layout_path=args.sensors,
        sensor_provenance_path=args.sensor_provenance,
        rainfall_provenance_path=args.rainfall_provenance,
        actuator_scope_path=args.actuator_scope,
        history_span_minutes=args.history_span_minutes,
        minimum_post_rain_tail_minutes=args.minimum_post_rain_tail_minutes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
