from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import canonical_baseline_id
from .code_contract import rtc_source_tree_sha256
from .inp_lineage import physical_contract_sha256, scientific_event_contract_sha256
from .inp_runtime import section_has_payload
from .pipeline import sha256_file
from .replay_peak import replay_exact_global_peak
from .rule_baselines import AUTO_RBC_CONTRACT, AUTO_RBC_SOURCE, EFD_CONTRACT, EFD_SOURCE

_FORMAL_STRATEGIES = {
    "proposed",
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
}


def _decision_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("decision log row must be a JSON object")
            rows.append(value)
    return rows


def _verify_strategy_execution(
    *, metadata: dict[str, object], metadata_path: Path, strategy: str
) -> dict[str, object]:
    inp = Path(str(metadata["inp_path"]))
    decisions = _decision_rows(metadata_path.parent / str(metadata["decision_file"]))
    controls = bool(section_has_payload(inp, "CONTROLS"))
    controller_present = bool(metadata.get("controller_present"))
    rule_contract: str | None = None

    if strategy == "internal_rtc":
        if not controls or controller_present or decisions:
            raise ValueError("Internal-RTC requires native controls, no Python controller and no writes")
    elif strategy == "no_control":
        if controls or controller_present or decisions:
            raise ValueError("No-control requires controls disabled, no Python controller and no writes")
    elif strategy in {"all_open", "all_closed"}:
        expected = 1.0 if strategy == "all_open" else 0.0
        source = "ALL_OPEN" if strategy == "all_open" else "ALL_CLOSED"
        if controls or not controller_present or not decisions:
            raise ValueError(f"{strategy} requires Python control on a controls-disabled runtime")
        for row in decisions:
            settings = row.get("settings")
            if row.get("source") != source or not isinstance(settings, dict) or not settings:
                raise ValueError(f"{strategy} decision log does not prove the requested strategy")
            if any(abs(float(v) - expected) > 1e-9 for v in settings.values()):
                raise ValueError(f"{strategy} did not command every recorded actuator to {expected}")
    elif strategy in {"auto_rbc", "efd"}:
        source = AUTO_RBC_SOURCE if strategy == "auto_rbc" else EFD_SOURCE
        rule_contract = AUTO_RBC_CONTRACT if strategy == "auto_rbc" else EFD_CONTRACT
        if controls or not controller_present or not decisions:
            raise ValueError(f"{strategy} requires causal Python decisions on controls-disabled SWMM")
        for row in decisions:
            settings = row.get("settings")
            diagnostics = row.get("diagnostics")
            if row.get("source") != source or not isinstance(settings, dict) or not settings:
                raise ValueError(f"{strategy} decision source/settings are invalid")
            if any(not 0.0 <= float(v) <= 1.0 for v in settings.values()):
                raise ValueError(f"{strategy} contains an out-of-range actuator setting")
            if not isinstance(diagnostics, dict) or diagnostics.get("rule_contract") != rule_contract:
                raise ValueError(f"{strategy} decision does not carry the frozen rule contract")
    elif strategy == "proposed":
        if controls or not controller_present or not decisions:
            raise ValueError("Proposed requires non-empty Python decisions on controls-disabled SWMM")
        if metadata.get("strategy") != "proposed":
            raise ValueError("Proposed metadata was not stamped by the production guard")
        if metadata.get("rtc_source_tree_sha256") != rtc_source_tree_sha256():
            raise ValueError("Proposed uses an incompatible implementation contract")
        for field in (
            "controller_config_sha256",
            "graph_schema_sha256",
            "step1_model_sha256",
            "step2_model_sha256",
        ):
            if not str(metadata.get(field, "")).strip():
                raise ValueError(f"Proposed lacks locked-artifact lineage: {field}")
    else:
        raise ValueError(f"unsupported Formal strategy: {strategy}")

    result: dict[str, object] = {
        "contract": "FORMAL_STRATEGY_EXECUTION_VERIFICATION_V1",
        "passed": True,
        "native_controls_enabled": controls,
        "python_controller_present": controller_present,
        "decision_count": len(decisions),
    }
    if rule_contract is not None:
        result["rule_contract"] = rule_contract
    return result


def formalize_run(
    *,
    main_metadata_path: str | Path,
    strategy: str,
    event_id: str,
    rainfall_group: str,
    output_path: str | Path,
) -> dict[str, object]:
    strategy = canonical_baseline_id(strategy)
    if strategy not in _FORMAL_STRATEGIES:
        raise ValueError(f"unsupported Formal strategy: {strategy}")
    meta_path = Path(main_metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("main run metadata must be a JSON object")
    if meta.get("exact_global_peak") is not False:
        raise ValueError("formal main run must use exact_global_peak=false")
    for key in ("inp_path", "inp_sha256", "decision_file", "node_statistics_file", "swmm_engine_version"):
        if key not in meta:
            raise ValueError(f"main run metadata lacks {key}")
    inp_path = Path(str(meta["inp_path"]))
    decision_path = meta_path.parent / str(meta["decision_file"])
    stats_path = meta_path.parent / str(meta["node_statistics_file"])
    if not inp_path.is_file() or not decision_path.is_file() or not stats_path.is_file():
        raise ValueError("main run evidence is missing")
    if sha256_file(inp_path) != str(meta["inp_sha256"]):
        raise ValueError("main run INP bytes changed after execution")
    if meta.get("strategy") is not None and canonical_baseline_id(str(meta["strategy"])) != strategy:
        raise ValueError("requested Formal strategy differs from stamped main run strategy")

    strategy_evidence = _verify_strategy_execution(
        metadata=meta, metadata_path=meta_path, strategy=strategy
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    replay_path = out.with_suffix(".peak_replay.json")
    replay = replay_exact_global_peak(
        inp_path=inp_path,
        decision_log_path=decision_path,
        output_path=replay_path,
        source_main_metadata_path=meta_path,
    )
    if replay["source_main_metadata_sha256"] != sha256_file(meta_path):
        raise RuntimeError("peak replay did not bind to the intended main metadata")
    if replay["decision_log_sha256"] != sha256_file(decision_path):
        raise RuntimeError("peak replay did not bind to the intended decision schedule")
    current_code = rtc_source_tree_sha256()
    if replay.get("rtc_source_tree_sha256") != current_code:
        raise RuntimeError("peak replay uses an incompatible implementation")
    main_engine = str(meta.get("swmm_engine_version", "")).strip()
    if str(replay.get("swmm_engine_version", "")).strip() != main_engine:
        raise RuntimeError("Global Peak replay SWMM engine differs from main run")

    payload: dict[str, object] = {
        "contract": "FORMAL_CLOSED_LOOP_RUN_MANIFEST_V5_EVENT_ENGINE_BOUND",
        "rtc_source_tree_sha256": current_code,
        "event_id": str(event_id),
        "rainfall_group": str(rainfall_group),
        "strategy": strategy,
        "strategy_execution": strategy_evidence,
        "main_metadata_path": str(meta_path.resolve()),
        "main_metadata_sha256": sha256_file(meta_path),
        "node_statistics_path": str(stats_path.resolve()),
        "node_statistics_sha256": sha256_file(stats_path),
        "decision_log_path": str(decision_path.resolve()),
        "decision_log_sha256": sha256_file(decision_path),
        "peak_replay_path": str(replay_path.resolve()),
        "peak_replay_sha256": sha256_file(replay_path),
        "inp_path": str(inp_path.resolve()),
        "full_inp_sha256": sha256_file(inp_path),
        "physical_network_sha256": physical_contract_sha256(inp_path),
        "scientific_event_sha256": scientific_event_contract_sha256(inp_path),
        "swmm_engine_version": main_engine,
        "model_step_seconds": int(meta["observation_update_seconds"]),
        "control_update_seconds": int(meta["control_update_seconds"]),
        "controller_config_sha256": meta.get("controller_config_sha256"),
        "graph_schema_sha256": meta.get("graph_schema_sha256"),
        "step1_model_sha256": meta.get("step1_model_sha256"),
        "step2_model_sha256": meta.get("step2_model_sha256"),
        "truth_contract": {
            "pfv_tfv": "SWMM_NODE_STATISTICS_CUMULATIVE_FROM_MAIN_CAUSAL_RUN",
            "global_peak": "ROUTING_STEP_REPLAY_OF_FROZEN_EXECUTED_DECISION_LOG",
            "event_binding": "FULL_SCIENTIFIC_EVENT_INP_EXCEPT_CONTROLS_AND_THREADS",
            "engine_binding": "MAIN_AND_PEAK_REPLAY_SAME_SWMM_ENGINE",
            "strategy_binding": "ACTUAL_CONTROLS_CONTROLLER_DECISION_LOG_SEMANTICS",
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and bind one causal Formal run")
    parser.add_argument("--main-metadata", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--rainfall-group", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(formalize_run(
        main_metadata_path=args.main_metadata,
        strategy=args.strategy,
        event_id=args.event_id,
        rainfall_group=args.rainfall_group,
        output_path=args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
