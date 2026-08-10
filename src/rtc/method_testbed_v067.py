from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .splits import assign_rainfall_group_splits


CONTRACT = "WUHAN_RTC_METHOD_TESTBED_V067"
RAINFALL_CONTRACT = "WUHAN_DB4201_T641_2020_CHICAGO_30_V1"
NETWORK_CONTRACT = "WUHAN_METHOD_TESTBED_NETWORK_V1"
DEFAULT_RETURN_PERIODS = (5, 10, 20, 50, 100)
DEFAULT_DURATIONS_MIN = (60, 120, 180, 240, 300, 360)
CHICAGO_R = 0.39
IDF_A0 = 9.686
IDF_P_COEFF = 0.887
IDF_B_MIN = 11.23
IDF_N = 0.658
RAIN_STEP_MIN = 5
DEFAULT_WARMUP_MIN = 60
DEFAULT_RECESSION_MIN = 360
DEFAULT_ORIFICE_TRAVEL_MIN = 10
KNOWN_NOOP_OFF_ACTUATORS = {
    "VP0600010.3",
    "VP0600010.4",
    "VP0600010.5",
    "add300.1",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sections(lines: list[str]) -> dict[str, tuple[int, int]]:
    starts: list[tuple[str, int]] = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            starts.append((s[1:-1].strip().upper(), i))
    result: dict[str, tuple[int, int]] = {}
    for j, (name, start) in enumerate(starts):
        end = starts[j + 1][1] if j + 1 < len(starts) else len(lines)
        result[name] = (start, end)
    return result


def _body_tokens(raw: str) -> list[str]:
    return raw.split(";", 1)[0].strip().split()


def _replace_section(lines: list[str], name: str, payload: list[str]) -> list[str]:
    sec = _sections(lines)
    if name.upper() not in sec:
        raise ValueError(f"INP lacks [{name}]")
    start, end = sec[name.upper()]
    return lines[: start + 1] + payload + [""] + lines[end:]


def _set_option(lines: list[str], key: str, value: str) -> None:
    sec = _sections(lines)
    start, end = sec["OPTIONS"]
    target = key.upper()
    for i in range(start + 1, end):
        tokens = _body_tokens(lines[i])
        if tokens and tokens[0].upper() == target:
            comment = ""
            if ";" in lines[i]:
                comment = " ;" + lines[i].split(";", 1)[1].strip()
            lines[i] = f"{key:<24}{value}{comment}".rstrip()
            return
    raise ValueError(f"[OPTIONS] lacks {key}")


def _parse_pump_curve_ids(lines: list[str]) -> set[str]:
    sec = _sections(lines)
    start, end = sec["PUMPS"]
    return {
        tokens[3]
        for raw in lines[start + 1 : end]
        if len(tokens := _body_tokens(raw)) >= 4
    }


def _modify_curves_to_variable_depth(lines: list[str], pump_curve_ids: set[str]) -> int:
    sec = _sections(lines)
    start, end = sec["CURVES"]
    changed = 0
    seen: set[str] = set()
    for i in range(start + 1, end):
        raw = lines[i]
        body, sep, comment = raw.partition(";")
        tokens = body.strip().split()
        if len(tokens) < 3 or tokens[0] not in pump_curve_ids or tokens[0] in seen:
            continue
        seen.add(tokens[0])
        if len(tokens) >= 4 and tokens[1].upper().startswith("PUMP"):
            if tokens[1].upper() == "PUMP2":
                tokens[1] = "PUMP4"
                changed += 1
            elif tokens[1].upper() != "PUMP4":
                raise ValueError(
                    f"pump curve {tokens[0]} has unexpected type {tokens[1]}; "
                    "v0.6.7 only migrates PUMP2 depth-flow curves to PUMP4"
                )
            rebuilt = "    ".join(tokens)
            if sep:
                rebuilt += " ;" + comment.strip()
            lines[i] = rebuilt
    return changed


def _modify_orifices(lines: list[str], *, travel_minutes: int) -> tuple[int, int]:
    if travel_minutes <= 0:
        raise ValueError("orifice travel time must be positive")
    travel_hours = travel_minutes / 60.0
    sec = _sections(lines)
    start, end = sec["ORIFICES"]
    changed = flap_changed = 0
    for i in range(start + 1, end):
        raw = lines[i]
        body, sep, comment = raw.partition(";")
        tokens = body.strip().split()
        if len(tokens) < 8:
            continue
        aid = tokens[0]
        if aid.startswith("RTC_IN_") or aid.startswith("RTC_OUT_"):
            if tokens[6].upper() != "YES":
                flap_changed += 1
            tokens[6] = "YES"
        tokens[7] = f"{travel_hours:.6f}".rstrip("0").rstrip(".")
        changed += 1
        rebuilt = "    ".join(tokens)
        if sep:
            rebuilt += " ;" + comment.strip()
        lines[i] = rebuilt
    return changed, flap_changed


def _repair_known_noop_rules(lines: list[str]) -> int:
    sec = _sections(lines)
    start, end = sec["CONTROLS"]
    repaired = 0
    active_rule = ""
    for i in range(start + 1, end):
        raw = lines[i]
        body, sep, comment = raw.partition(";")
        tokens = body.strip().split()
        if not tokens:
            continue
        if tokens[0].upper() == "RULE" and len(tokens) >= 2:
            active_rule = tokens[1]
            continue
        if not active_rule.lower().endswith("_off"):
            continue
        if len(tokens) >= 6 and tokens[0].upper() in {"THEN", "AND"}:
            aid = tokens[2]
            if (
                aid in KNOWN_NOOP_OFF_ACTUATORS
                and tokens[3].upper() == "SETTING"
                and tokens[4] == "="
                and float(tokens[5]) == 1.0
            ):
                tokens[5] = "0"
                repaired += 1
                rebuilt = " ".join(tokens)
                if sep:
                    rebuilt += " ;" + comment.strip()
                lines[i] = rebuilt
    return repaired


def _count_section_payload(lines: list[str], section: str) -> int:
    sec = _sections(lines)
    if section not in sec:
        return 0
    start, end = sec[section]
    return sum(bool(_body_tokens(x)) for x in lines[start + 1 : end])


def _outfall_types(lines: list[str]) -> list[str]:
    sec = _sections(lines)
    start, end = sec["OUTFALLS"]
    values = []
    for raw in lines[start + 1 : end]:
        tokens = _body_tokens(raw)
        if len(tokens) >= 3:
            values.append(tokens[2].upper())
    return values


def build_method_testbed_network(
    source: str | Path,
    destination: str | Path,
    *,
    orifice_travel_minutes: int = DEFAULT_ORIFICE_TRAVEL_MIN,
) -> dict[str, object]:
    src = Path(source).resolve()
    dst = Path(destination)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    source_sha = sha256_file(src)
    pump_curves = _parse_pump_curve_ids(lines)
    pumps_changed = _modify_curves_to_variable_depth(lines, pump_curves)
    orifices_changed, flap_changed = _modify_orifices(
        lines, travel_minutes=orifice_travel_minutes
    )
    controls_repaired = _repair_known_noop_rules(lines)

    outfall_types = _outfall_types(lines)
    if not outfall_types or any(x != "FREE" for x in outfall_types):
        raise ValueError("v0.6.7 method-testbed contract requires all source outfalls to remain FREE")
    if _count_section_payload(lines, "DWF") == 0:
        raise ValueError("v0.6.7 method-testbed source must contain the supplied idealized DWF")

    dst.parent.mkdir(parents=True, exist_ok=True)
    preamble = [
        ";; WUHAN RTC METHOD TESTBED v0.6.7",
        ";; Scientific scope: idealized SWMM method evaluation; not a field digital twin.",
        ";; DWF retained as idealized background hydraulics; all OUTFALLS remain FREE.",
        ";; Pump PUMP2 depth-flow curves migrated to PUMP4 for continuous depth-dependent operation.",
        f";; All orifices use {orifice_travel_minutes}-min full travel time; RTC storage IN/OUT links are flap-gated.",
        ";; Four known copied OFF rules that commanded SETTING=1 were repaired to SETTING=0.",
        "",
    ]
    dst.write_text("\n".join(preamble + lines) + "\n", encoding="utf-8")
    return {
        "contract": NETWORK_CONTRACT,
        "source_path": str(src),
        "source_sha256": source_sha,
        "output_path": str(dst.resolve()),
        "output_sha256": sha256_file(dst),
        "pump_curves_pump2_to_pump4": pumps_changed,
        "orifices_with_travel_time": orifices_changed,
        "rtc_flap_gates_enabled": flap_changed,
        "native_noop_off_rules_repaired": controls_repaired,
        "dwf_entries_preserved": _count_section_payload(lines, "DWF"),
        "outfalls_free_preserved": len(outfall_types),
        "subareas_modified": False,
        "field_digital_twin_claim": False,
    }


def wuhan_idf_average_mm_min(return_period_year: float, duration_min: float) -> float:
    if not (2 <= return_period_year <= 100):
        raise ValueError("DB4201/T 641-2020 return period must be 2..100 years")
    if not (5 <= duration_min <= 1440):
        raise ValueError("DB4201/T 641-2020 duration must be 5..1440 min")
    a = IDF_A0 * (1.0 + IDF_P_COEFF * math.log10(return_period_year))
    return a / ((duration_min + IDF_B_MIN) ** IDF_N)


def _chicago_antiderivative(distance_min: float, fraction: float, a: float) -> float:
    if distance_min <= 0:
        return 0.0
    u = distance_min / fraction
    return a * distance_min / ((u + IDF_B_MIN) ** IDF_N)


def chicago_5min_hyetograph(
    return_period_year: int,
    duration_min: int,
    *,
    r: float = CHICAGO_R,
    step_min: int = RAIN_STEP_MIN,
) -> list[tuple[int, float]]:
    if duration_min % step_min:
        raise ValueError("duration must be an integer multiple of rainfall step")
    if not 0 < r < 1:
        raise ValueError("Chicago peak fraction must be in (0,1)")
    a = IDF_A0 * (1.0 + IDF_P_COEFF * math.log10(return_period_year))
    peak = r * duration_min

    def block_depth(t0: float, t1: float) -> float:
        depth = 0.0
        if t0 < peak:
            left = t0
            right = min(t1, peak)
            if right > left:
                d0 = peak - left
                d1 = peak - right
                depth += _chicago_antiderivative(d0, r, a) - _chicago_antiderivative(d1, r, a)
        if t1 > peak:
            left = max(t0, peak)
            right = t1
            if right > left:
                d0 = left - peak
                d1 = right - peak
                depth += (
                    _chicago_antiderivative(d1, 1.0 - r, a)
                    - _chicago_antiderivative(d0, 1.0 - r, a)
                )
        return max(0.0, depth)

    result: list[tuple[int, float]] = []
    for minute in range(0, duration_min, step_min):
        depth = block_depth(minute, minute + step_min)
        result.append((minute, depth / step_min * 60.0))
    analytic = wuhan_idf_average_mm_min(return_period_year, duration_min) * duration_min
    discrete = sum(v * step_min / 60.0 for _, v in result)
    if not math.isclose(analytic, discrete, rel_tol=2e-10, abs_tol=2e-10):
        raise RuntimeError(f"Chicago depth integration mismatch: {discrete} != {analytic}")
    return result


def _format_swmm_datetime(dt: datetime) -> tuple[str, str]:
    return dt.strftime("%m/%d/%Y"), dt.strftime("%H:%M:%S")


def _build_event_lines(
    base_lines: list[str],
    *,
    return_period_year: int,
    duration_min: int,
    warmup_minutes: int,
    recession_minutes: int,
) -> tuple[list[str], list[tuple[int, float]]]:
    lines = list(base_lines)
    storm_start = datetime(2022, 8, 11, 0, 0, 0)
    sim_start = storm_start - timedelta(minutes=warmup_minutes)
    storm_end = storm_start + timedelta(minutes=duration_min)
    sim_end = storm_end + timedelta(minutes=recession_minutes)
    for key, value in (
        ("START_DATE", _format_swmm_datetime(sim_start)[0]),
        ("START_TIME", _format_swmm_datetime(sim_start)[1]),
        ("REPORT_START_DATE", _format_swmm_datetime(sim_start)[0]),
        ("REPORT_START_TIME", _format_swmm_datetime(sim_start)[1]),
        ("END_DATE", _format_swmm_datetime(sim_end)[0]),
        ("END_TIME", _format_swmm_datetime(sim_end)[1]),
    ):
        _set_option(lines, key, value)

    lines = _replace_section(
        lines,
        "RAINGAGES",
        [
            ";;Name                 Format    Interval SCF Source",
            "RG_WUHAN_DESIGN        INTENSITY 0:05     1.0 TIMESERIES TS_WUHAN_DESIGN",
        ],
    )

    sec = _sections(lines)
    start, end = sec["SUBCATCHMENTS"]
    for i in range(start + 1, end):
        raw = lines[i]
        body, sep, comment = raw.partition(";")
        tokens = body.strip().split()
        if len(tokens) >= 2:
            tokens[1] = "RG_WUHAN_DESIGN"
            rebuilt = "    ".join(tokens)
            if sep:
                rebuilt += " ;" + comment.strip()
            lines[i] = rebuilt

    hyeto = chicago_5min_hyetograph(return_period_year, duration_min)
    ts_payload = [";;Name                 Date       Time       Value"]
    for minute, intensity in hyeto:
        stamp = storm_start + timedelta(minutes=minute)
        date, time = _format_swmm_datetime(stamp)
        ts_payload.append(f"TS_WUHAN_DESIGN         {date} {time} {intensity:.8f}")
    stamp = storm_start + timedelta(minutes=duration_min)
    date, time = _format_swmm_datetime(stamp)
    ts_payload.append(f"TS_WUHAN_DESIGN         {date} {time} 0.00000000")
    lines = _replace_section(lines, "TIMESERIES", ts_payload)
    return lines, hyeto


def generate_30_event_library(
    network_inp: str | Path,
    output_root: str | Path,
    *,
    return_periods: tuple[int, ...] = DEFAULT_RETURN_PERIODS,
    durations_min: tuple[int, ...] = DEFAULT_DURATIONS_MIN,
    warmup_minutes: int = DEFAULT_WARMUP_MIN,
    recession_minutes: int = DEFAULT_RECESSION_MIN,
    split_seed: int = 42,
) -> dict[str, object]:
    if warmup_minutes < 60:
        raise ValueError("warm-up must be >= 60 min to supply the default 13-frame causal history")
    network = Path(network_inp).resolve()
    root = Path(output_root)
    events_dir = root / "events"
    rain_dir = root / "rainfall"
    contracts_dir = root / "contracts"
    for directory in (events_dir, rain_dir, contracts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base = network.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, object]] = []
    for rp in return_periods:
        for duration in durations_min:
            event_id = f"T{rp}_D{duration}_chicago"
            lines, hyeto = _build_event_lines(
                base,
                return_period_year=rp,
                duration_min=duration,
                warmup_minutes=warmup_minutes,
                recession_minutes=recession_minutes,
            )
            event_path = events_dir / f"{event_id}.inp"
            event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rain_path = rain_dir / f"{event_id}.csv"
            with rain_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["elapsed_min", "intensity_mmhr"])
                writer.writerows(hyeto)
                writer.writerow([duration, 0.0])
            total_depth = sum(v * RAIN_STEP_MIN / 60.0 for _, v in hyeto)
            rows.append(
                {
                    "event_id": event_id,
                    "rainfall_group": event_id,
                    "return_period_year": rp,
                    "duration_minutes": duration,
                    "pattern": "chicago_r0.39",
                    "total_depth_mm": total_depth,
                    "peak_intensity_mmhr": max(v for _, v in hyeto),
                    "antecedent_rainfall_mm": 0.0,
                    "pre_rain_warmup_minutes": warmup_minutes,
                    "post_rain_tail_minutes": recession_minutes,
                    "simulation_duration_minutes": warmup_minutes + duration + recession_minutes,
                    "inp_path": str(event_path.resolve()),
                    "rainfall_csv_path": str(rain_path.resolve()),
                    "prepared_inp_sha256": sha256_file(event_path),
                    "rainfall_sha256": sha256_file(rain_path),
                    "event_preparation_contract": CONTRACT,
                    "rainfall_contract": RAINFALL_CONTRACT,
                }
            )

    frame = assign_rainfall_group_splits(pd.DataFrame(rows), seed=split_seed)
    source_manifest = contracts_dir / "source_event_manifest.csv"
    split_registry = contracts_dir / "events_with_splits.csv"
    pd.DataFrame(rows).to_csv(source_manifest, index=False)
    frame.to_csv(split_registry, index=False)
    rainfall_provenance = {
        "contract": "RAINFALL_PROVENANCE_V1",
        "source_kind": "design_storm_formula",
        "official_standard_claim": True,
        "official_standard": "DB4201/T 641-2020",
        "idf_formula_mm_min": "9.686*(1+0.887*log10(P))/(t+11.23)^0.658",
        "idf_applicability": {"return_period_years": [2, 100], "duration_minutes": [5, 1440]},
        "chicago_peak_fraction_r": CHICAGO_R,
        "rainfall_step_minutes": RAIN_STEP_MIN,
        "spatial_mode": "uniform_single_design_gage_across_all_subcatchments",
        "return_period_scope_years": list(return_periods),
        "duration_scope_minutes": list(durations_min),
        "pattern_scope": ["chicago_r0.39"],
        "event_count": len(frame),
        "methodology_testbed_only": True,
        "historical_observed_rainfall_claim": False,
    }
    (contracts_dir / "rainfall_provenance.v067.json").write_text(
        json.dumps(rainfall_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "contract": RAINFALL_CONTRACT,
        "events": len(frame),
        "return_periods": list(return_periods),
        "durations_min": list(durations_min),
        "chicago_r": CHICAGO_R,
        "warmup_minutes": warmup_minutes,
        "recession_minutes": recession_minutes,
        "fixed_evaluation_window_not_full_recovery_claim": True,
        "network_inp_sha256": sha256_file(network),
        "registry": str(split_registry.resolve()),
        "registry_sha256": sha256_file(split_registry),
    }
    (contracts_dir / "rainfall_library_summary.v067.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_fresh_project7_inputs(
    *,
    source_inp: str | Path,
    sensor_nodes: str | Path,
    priority_nodes: str | Path,
    output_root: str | Path,
    orifice_travel_minutes: int = DEFAULT_ORIFICE_TRAVEL_MIN,
    warmup_minutes: int = DEFAULT_WARMUP_MIN,
    recession_minutes: int = DEFAULT_RECESSION_MIN,
) -> dict[str, object]:
    root = Path(output_root)
    network_dir = root / "network"
    contracts_dir = root / "contracts"
    network_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)
    network_path = network_dir / "wuhan_method_testbed_v067.inp"
    network_evidence = build_method_testbed_network(
        source_inp, network_path, orifice_travel_minutes=orifice_travel_minutes
    )
    rain_evidence = generate_30_event_library(
        network_path,
        root,
        warmup_minutes=warmup_minutes,
        recession_minutes=recession_minutes,
    )
    sensors_dst = contracts_dir / "sensor_nodes.txt"
    priority_dst = contracts_dir / "priority_nodes.txt"
    shutil.copyfile(sensor_nodes, sensors_dst)
    shutil.copyfile(priority_nodes, priority_dst)
    actuator_scope = {
        "contract": "ACTUATOR_SCOPE_V1",
        "actuation_scope": "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY",
        "field_deployment_claim": False,
        "actuator_count": 109,
        "continuous_setting_range": [0.0, 1.0],
        "max_setting_delta_per_10min_update": 0.5,
        "pump_semantics": "PUMP4 depth-flow curves; SWMM setting is a continuous flow multiplier; methodology-testbed assumption",
        "orifice_semantics": f"continuous fractional opening; {orifice_travel_minutes}-min full travel time",
        "retrofit_storage_links": "RTC_IN_*/RTC_OUT_* are direction-specific and flap-gated",
    }
    (contracts_dir / "actuator_scope.v067.json").write_text(
        json.dumps(actuator_scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    method_contract = {
        "contract": CONTRACT,
        "objective": "minimize system-wide cumulative TFV",
        "pfv_role": "soft_secondary_diagnostic",
        "global_peak_role": "report_only",
        "claim_scope": "methodology test on an idealized simplified Wuhan SWMM; reduce sewer-node overflow",
        "outfalls": "preserve all source FREE outfalls",
        "dwf": "retain supplied idealized DWF as background hydraulic loading",
        "subareas": "preserved unchanged",
        "pre_rain_warmup_minutes": warmup_minutes,
        "post_rain_evaluation_tail_minutes": recession_minutes,
        "recovery_claim": False,
        "fixed_window_tf_volume_claim": True,
        "rainfall_contract": RAINFALL_CONTRACT,
        "network_contract": NETWORK_CONTRACT,
        "sensor_layout_sha256": sha256_file(sensors_dst),
        "priority_nodes_sha256": sha256_file(priority_dst),
        "network_sha256": sha256_file(network_path),
    }
    (contracts_dir / "method_testbed_contract.v067.json").write_text(
        json.dumps(method_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "contract": CONTRACT,
        "output_root": str(root.resolve()),
        "network": network_evidence,
        "rainfall": rain_evidence,
        "sensors": str(sensors_dst.resolve()),
        "priority": str(priority_dst.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v0.6.7 Wuhan TFV-first methodology-testbed inputs from source-only assets"
    )
    parser.add_argument("--source-inp", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--orifice-travel-minutes", type=int, default=DEFAULT_ORIFICE_TRAVEL_MIN)
    parser.add_argument("--warmup-minutes", type=int, default=DEFAULT_WARMUP_MIN)
    parser.add_argument("--recession-minutes", type=int, default=DEFAULT_RECESSION_MIN)
    args = parser.parse_args()
    payload = build_fresh_project7_inputs(
        source_inp=args.source_inp,
        sensor_nodes=args.sensors,
        priority_nodes=args.priority,
        output_root=args.out_root,
        orifice_travel_minutes=args.orifice_travel_minutes,
        warmup_minutes=args.warmup_minutes,
        recession_minutes=args.recession_minutes,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
