from __future__ import annotations

import json
import math
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


def _main_callback_stride(metadata_path: str | Path | None) -> int | None:
    """Return the Python callback stride used by the original causal main run."""

    if metadata_path is None:
        return None
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("source main metadata must be a JSON object")
    values = [
        int(meta["control_update_seconds"]),
        int(meta["observation_update_seconds"]),
        int(meta["record_stride_seconds"]),
    ]
    if any(value <= 0 for value in values):
        raise ValueError("source main metadata contains invalid callback cadence")
    return math.gcd(math.gcd(values[0], values[1]), values[2])


def replay_exact_global_peak(
    *,
    inp_path: str | Path,
    decision_log_path: str | Path | None,
    output_path: str | Path,
    source_main_metadata_path: str | Path | None = None,
) -> dict[str, object]:
    """Replay the frozen executed policy while observing flooding every routing callback.

    The key distinction is observation cadence versus control-write cadence. The replay sees
    every SWMM routing callback to obtain the exact synchronous Global Peak, but Python
    actuator targets are reasserted only on the same callback grid as the original main run.
    This avoids changing intrinsic pump/local hydraulic behaviour between supervisory
    callbacks merely because reporting is sampled more frequently.
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

    callback_stride = _main_callback_stride(source_main_metadata_path)
    if schedule and callback_stride is None:
        raise ValueError(
            "decision replay requires source main metadata so the original Python write cadence is preserved"
        )

    peak = 0.0
    applied = 0
    last_elapsed = -1
    next_python_callback = callback_stride if callback_stride is not None else None
    held_settings: dict[str, float] | None = None
    flow_units = ""
    engine_version = ""
    flow_error = float("nan")
    python_reassertions = 0
    try:
        with Simulation(
            str(inp),
            reportfile=str(report_path),
            outputfile=str(engine_output_path),
        ) as sim:
            # No step_advance(): Python observes every SWMM routing callback for peak only.
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

                # Sample the same pre-write hydraulic state that the causal main callback sees.
                total_native = sum(
                    max(0.0, float(obj.flooding)) for obj in node_obj.values()
                )
                peak = max(
                    peak, float(flow_rate_to_m3s(total_native, flow_units))
                )

                python_callback_due = (
                    next_python_callback is not None
                    and elapsed >= next_python_callback
                )
                if not python_callback_due:
                    continue

                new_decision = False
                while (
                    applied < len(schedule)
                    and int(schedule[applied]["elapsed_seconds"]) <= elapsed
                ):
                    decision_elapsed = int(schedule[applied]["elapsed_seconds"])
                    if callback_stride is not None and decision_elapsed % callback_stride:
                        raise ValueError(
                            "decision log contains a time off the original Python callback grid"
                        )
                    settings = schedule[applied]["settings"]
                    assert isinstance(settings, dict)
                    held_settings = {
                        aid: float(settings[aid]) for aid in catalog.ids
                    }
                    applied += 1
                    new_decision = True

                # The original main run writes a new decision when due; otherwise it reasserts
                # the held supervisory target once per Python callback, not once per routing step.
                if held_settings is not None:
                    for aid, value in held_settings.items():
                        link_obj[aid].target_setting = value
                    if not new_decision:
                        python_reassertions += 1

                assert next_python_callback is not None and callback_stride is not None
                while next_python_callback <= elapsed:
                    next_python_callback += callback_stride
            flow_error = float(sim.flow_routing_error)
    finally:
        report_path.unlink(missing_ok=True)
        engine_output_path.unlink(missing_ok=True)

    if applied != len(schedule):
        raise RuntimeError(
            f"replay ended before all decisions were applied: {applied}/{len(schedule)}"
        )
    payload: dict[str, object] = {
        "contract": "ROUTING_STEP_GLOBAL_PEAK_REPLAY_V3_WRITE_CADENCE_PRESERVED",
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
        "main_python_callback_stride_seconds": callback_stride,
        "held_setting_reassertions": python_reassertions,
        "control_write_cadence_preserved": True,
        "flow_routing_error_pct": flow_error,
        "flow_units": flow_units,
        "swmm_engine_version": engine_version,
        "engine_files_retained": False,
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
