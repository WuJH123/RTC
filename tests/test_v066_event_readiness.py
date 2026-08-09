from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rtc.control_lineage import section_payload_sha256
from rtc.event_preparation import EVENT_PREPARATION_CONTRACT, prepare_event_registry
from rtc.inp_runtime import build_runtime_inp, section_has_payload, sha256_file
from rtc.study_readiness import validate_pretraining_readiness


NETWORK = """[OPTIONS]
FLOW_UNITS CMS
FLOW_ROUTING DYNWAVE
START_DATE 08/11/2022
START_TIME 00:00:00
REPORT_START_DATE 08/11/2022
REPORT_START_TIME 00:00:00
END_DATE 08/11/2022
END_TIME 04:00:00
REPORT_STEP 00:05:00
WET_STEP 00:05:00
DRY_STEP 00:05:00
ROUTING_STEP 00:00:15
RULE_STEP 00:00:10
THREADS 1

[RAINGAGES]
RG INTENSITY 0:05 1.0 TIMESERIES RAIN

[JUNCTIONS]
J1 0 2 0 0 0
J2 0 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[SUBCATCHMENTS]
S1 RG J1 1 50 100 0 0

[SUBAREAS]
S1 0.01 0.1 1 5 0 OUTLET 100

[INFILTRATION]
S1 1 1 1 1 1

[PUMPS]
P1 J1 J2 * OFF 0 0

[ORIFICES]
G1 J2 O1 SIDE 0 0.5 NO 0

[TIMESERIES]
RAIN 00:00 10
RAIN 00:05 5
RAIN 00:10 0

[DWF]
J1 FLOW 0.01

[CONTROLS]
RULE P1_ON
IF NODE J1 DEPTH > 0.5
THEN PUMP P1 STATUS = ON
"""


def _event_text() -> str:
    return NETWORK.replace(
        "RULE P1_ON\nIF NODE J1 DEPTH > 0.5\nTHEN PUMP P1 STATUS = ON\n", ""
    )


def test_internal_runtime_pairs_event_dwf_and_template_rules(tmp_path: Path) -> None:
    event = tmp_path / "event.inp"
    template = tmp_path / "network.inp"
    runtime = tmp_path / "runtime.inp"
    event.write_text(_event_text(), encoding="utf-8")
    template.write_text(NETWORK, encoding="utf-8")

    contract = build_runtime_inp(
        event,
        runtime,
        native_controls=True,
        swmm_threads=1,
        native_controls_template=template,
    )
    assert section_has_payload(runtime, "CONTROLS")
    assert "J1 FLOW 0.01" in runtime.read_text(encoding="utf-8")
    assert section_payload_sha256(runtime, "CONTROLS") == section_payload_sha256(
        template, "CONTROLS"
    )
    assert contract.native_controls_template_sha256 == sha256_file(template)


def test_native_controls_without_source_or_template_fails_closed(tmp_path: Path) -> None:
    event = tmp_path / "event.inp"
    runtime = tmp_path / "runtime.inp"
    event.write_text(_event_text(), encoding="utf-8")
    try:
        build_runtime_inp(event, runtime, native_controls=True, swmm_threads=1)
    except ValueError as exc:
        assert "no executable [CONTROLS]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("native_controls=True must not silently pass an empty event rule set")


def test_event_preparation_creates_dry_prefix_and_recovery_tail(tmp_path: Path) -> None:
    source = tmp_path / "event.inp"
    source.write_text(_event_text(), encoding="utf-8")
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "G1",
                "inp_path": str(source),
                "scientific_split": "development",
                "development_fold": "train",
            }
        ]
    )
    prepared = prepare_event_registry(
        events,
        output_dir=tmp_path / "prepared",
        warmup_minutes=60,
        post_rain_tail_minutes=360,
    )
    row = prepared.iloc[0]
    assert row["event_preparation_contract"] == EVENT_PREPARATION_CONTRACT
    assert float(row["rainfall_onset_elapsed_minutes"]) == 60.0
    assert float(row["pre_rain_warmup_minutes"]) == 60.0
    assert int(row["post_rain_tail_minutes"]) == 360
    text = Path(str(row["inp_path"])).read_text(encoding="utf-8")
    assert "START_DATE          08/10/2022" in text
    assert "START_TIME          23:00:00" in text
    assert "RAIN                08/11/2022 00:00:00 10" in text
    assert "END_DATE            08/11/2022" in text
    assert "END_TIME            06:10:00" in text


def test_study_readiness_accepts_explicit_simulation_only_scope(tmp_path: Path) -> None:
    source = tmp_path / "event.inp"
    frozen = tmp_path / "network.inp"
    sensors = tmp_path / "sensors.txt"
    source.write_text(_event_text(), encoding="utf-8")
    frozen.write_text(NETWORK, encoding="utf-8")
    sensors.write_text("J1\n", encoding="utf-8")
    events = prepare_event_registry(
        pd.DataFrame(
            [
                {
                    "event_id": "E1",
                    "rainfall_group": "G1",
                    "inp_path": str(source),
                    "scientific_split": "development",
                    "development_fold": "train",
                }
            ]
        ),
        output_dir=tmp_path / "prepared",
        warmup_minutes=60,
        post_rain_tail_minutes=360,
    )
    registry = tmp_path / "events.csv"
    events.to_csv(registry, index=False)

    sensor_prov = tmp_path / "sensor.json"
    sensor_prov.write_text(
        json.dumps(
            {
                "contract": "SENSOR_LAYOUT_PROVENANCE_V1",
                "sensor_layout_sha256": sha256_file(sensors),
                "hydraulic_outcomes_used_for_selection": False,
                "method": "topology only",
            }
        ),
        encoding="utf-8",
    )
    rain_prov = tmp_path / "rain.json"
    rain_prov.write_text(
        json.dumps(
            {
                "contract": "RAINFALL_PROVENANCE_V1",
                "source_kind": "test design storm",
                "official_standard_claim": False,
                "spatial_mode": "UNIFORM",
                "return_period_scope_years": [10],
                "duration_scope_minutes": [10],
                "pattern_scope": ["block"],
            }
        ),
        encoding="utf-8",
    )
    actuator = tmp_path / "actuator.json"
    actuator.write_text(
        json.dumps(
            {
                "contract": "ACTUATOR_SCOPE_V1",
                "actuation_scope": "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY",
                "actuator_count": 2,
                "field_deployment_claim": False,
            }
        ),
        encoding="utf-8",
    )

    result = validate_pretraining_readiness(
        event_registry_path=registry,
        frozen_inp_path=frozen,
        sensor_layout_path=sensors,
        sensor_provenance_path=sensor_prov,
        rainfall_provenance_path=rain_prov,
        actuator_scope_path=actuator,
        history_span_minutes=60,
        minimum_post_rain_tail_minutes=360,
    )
    assert result["passed"] is True
    assert result["actuation_scope"] == "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY"
    assert result["field_deployment_claim"] is False
