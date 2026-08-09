from __future__ import annotations

import json
from pathlib import Path

from .code_contract import rtc_source_tree_sha256
from .inp import discover_actuators, discover_nodes
from .inp_runtime import sha256_file
from .units import flow_rate_to_m3s


def _decision_schedule(path: str | Path) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            item = json.loads(raw)
            if "elapsed_seconds" not in item or "settings" not in item:
                raise ValueError(f"invalid decision-log row in {path}")
            decisions.append(item)
    decisions.sort(key=lambda x: int(x["elapsed_seconds"]))
    if any(
        int(decisions[i]["elapsed_seconds"])
        >= int(decisions[i + 1]["elapsed_seconds"])
        for i in range(len(decisions) - 1)
    ):
        raise ValueError("decision log times must be strictly increasing")
    return decisions


def replay_exact_global_peak(
    *,
    inp_path: str | Path,
    decision_log_path: str | Path | None,
    output_path: str | Path,
    source_main_metadata_path: str | Path | None = None,
) -> dict[str, object]:
    """Replay one frozen policy at every SWMM routing callback for Global Peak only.

    No controller/model/forecast is invoked. Unique report/output files are placed next to
    the requested replay evidence and deleted in ``finally`` so parallel Final workers never
    contend for SWMM defaults and successful metric replay leaves no large engine artefacts.
    """

    try:
        from pyswmm import Links, Nodes, Simulation
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the SWMM extra: pip install -e '.[swmm]'") from exc

    inp = Path(inp_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = out.with_suffix(out.suffix + ".peak_replay.rpt")
    engine_output_path = out.with_suffix(out.suffix + ".peak_replay.out")
    # A stale file from a killed earlier replay is not evidence and must never be reused.
    report_path.unlink(missing_ok=True)
    engine_output_path.unlink(missing_ok=True)

    catalog = discover_actuators(inp)
    node_ids = discover_nodes(inp)
    schedule = _decision_schedule(decision_log_path) if decision_log_path else []
    expected = set(catalog.ids)
    for item in schedule:
        settings = item["settings"]
        if not isinstance(settings, dict) or set(settings) != expected:
            raise ValueError(
                "replay decision does not specify the complete frozen actuator set"
            )
        if any(not 0.0 <= float(value) <= 1.0 for value in settings.values()):
            raise ValueError("replay decision contains setting outside [0,1]")

    peak = 0.0
    applied = 0
    last_elapsed = -1
    flow_units = ""
    engine_version = ""
    flow_error = float("nan")
    try:
        with Simulation(
            str(inp),
            reportfile=str(report_path),
            outputfile=str(engine_output_path),
        ) as sim:
            # Deliberately do not call step_advance(): Python observes each routing callback.
            flow_units = str(sim.flow_units)
            engine_version = str(sim.engine_version)
            links, nodes = Links(sim), Nodes(sim)
            link_obj = {aid: links[aid] for aid in catalog.ids}
            node_obj = {nid: nodes[nid] for nid in node_ids}
            for _ in sim:
                elapsed = int((sim.current_time - sim.start_time).total_seconds())
                if elapsed < last_elapsed:
                    raise RuntimeError("SWMM replay time moved backwards")
                last_elapsed = elapsed
                while (
                    applied < len(schedule)
                    and int(schedule[applied]["elapsed_seconds"]) <= elapsed
                ):
                    settings = schedule[applied]["settings"]
                    assert isinstance(settings, dict)
                    for aid in catalog.ids:
                        link_obj[aid].target_setting = float(settings[aid])
                    applied += 1
                if applied:
                    # Re-assert the exact most recently executed Python action at each
                    # routing callback. Proposed/static Python policies use a controls-
                    # disabled runtime; Internal-RTC has an empty schedule and its native
                    # rules remain authoritative.
                    settings = schedule[applied - 1]["settings"]
                    assert isinstance(settings, dict)
                    for aid in catalog.ids:
                        link_obj[aid].target_setting = float(settings[aid])
                total_native = sum(
                    max(0.0, float(obj.flooding)) for obj in node_obj.values()
                )
                peak = max(
                    peak, float(flow_rate_to_m3s(total_native, flow_units))
                )
            flow_error = float(sim.flow_routing_error)
    finally:
        report_path.unlink(missing_ok=True)
        engine_output_path.unlink(missing_ok=True)

    if applied != len(schedule):
        raise RuntimeError(
            f"replay ended before all decisions were applied: {applied}/{len(schedule)}"
        )
    payload: dict[str, object] = {
        "contract": "ROUTING_STEP_GLOBAL_PEAK_REPLAY_V2_ISOLATED",
        "inp_path": str(inp.resolve()),
        "inp_sha256": sha256_file(inp),
        "decision_log_path": (
            str(Path(decision_log_path).resolve()) if decision_log_path else None
        ),
        "decision_log_sha256": (
            sha256_file(decision_log_path) if decision_log_path else None
        ),
        "source_main_metadata_path": (
            str(Path(source_main_metadata_path).resolve())
            if source_main_metadata_path
            else None
        ),
        "source_main_metadata_sha256": (
            sha256_file(source_main_metadata_path)
            if source_main_metadata_path
            else None
        ),
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "routing_step_global_peak_flood_rate_m3s": float(peak),
        "decisions_applied": applied,
        "flow_routing_error_pct": flow_error,
        "flow_units": flow_units,
        "swmm_engine_version": engine_version,
        "engine_files_retained": False,
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
