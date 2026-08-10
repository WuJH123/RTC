from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import policy_lock as legacy
from .code_contract import rtc_source_tree_sha256
from .control_lineage import section_payload_sha256
from .inp_lineage import physical_contract_sha256
from .project7_contract import PRODUCTION_CONTROLLER_CONTRACT, validate_project7_runtime_config
from .study_readiness import READINESS_CONTRACT
from .tfv_pipeline import TFVPipelineLedger, sha256_file

POLICY_LOCK_CONTRACT = legacy.POLICY_LOCK_CONTRACT
EXPECTED_STRATEGIES = (
    "proposed",
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
)


def _verify_runtime_acceptance_v2(
    path: str | Path,
    *,
    controller_config_path: str | Path,
    control_update_seconds: int,
    implementation_sha: str,
) -> dict[str, object]:
    evidence = legacy._json(path)
    if evidence.get("contract") != "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY":
        raise ValueError("Policy Lock requires temporal-continuity runtime acceptance V2")
    if evidence.get("passed") is not True:
        raise ValueError("development runtime/continuity acceptance did not pass")
    if evidence.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError("runtime acceptance uses an incompatible implementation contract")
    if str(evidence.get("controller_config_sha256", "")) != sha256_file(
        controller_config_path
    ):
        raise ValueError("runtime acceptance was not generated with the locked controller config")
    budget = float(evidence.get("decision_runtime_budget_seconds", -1.0))
    if not 0 < budget < control_update_seconds:
        raise ValueError("runtime acceptance contains an invalid decision compute budget")
    metrics = evidence.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("runtime acceptance lacks metrics")
    zero_required = (
        "control_grid_violations",
        "first_decision_violations",
        "missing_runtime_diagnostics",
        "continuity_evidence_missing",
        "cross_decision_continuity_violations",
        "planned_horizon_continuity_violations",
        "fatal_runtime_fallbacks",
    )
    for key in zero_required:
        if int(metrics.get(key, -1)) != 0:
            raise ValueError(f"runtime acceptance contains nonzero {key}")
    if float(metrics.get("decision_runtime_max_seconds", float("inf"))) > budget:
        raise ValueError("development decision runtime exceeded the locked budget")
    if not isinstance(evidence.get("project7_runtime_contract"), dict):
        raise ValueError("runtime acceptance lacks Project7 runtime-contract evidence")
    return evidence


def create_policy_lock(
    *, ledger_path: str | Path, artefacts_path: str | Path, output_path: str | Path
) -> dict[str, object]:
    ledger = TFVPipelineLedger.from_json(ledger_path)
    ledger.require_ready_for_lock()
    raw = legacy._json(artefacts_path)
    artefacts = {str(k): str(v) for k, v in raw.items()}
    missing = sorted(legacy._REQUIRED - set(artefacts))
    if missing:
        raise ValueError(f"Policy Lock missing required artifacts: {missing}")
    if "study_readiness" not in artefacts:
        raise ValueError("Policy Lock requires the pretraining study_readiness artifact")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"Policy Lock artifact missing: {name}: {path}")

    implementation_sha = rtc_source_tree_sha256()
    workspace, rainfall_design = legacy._verify_workspace_and_split(artefacts)
    preflight = legacy._json(artefacts["inp_preflight"])
    if preflight.get("contract") != "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC":
        raise ValueError("frozen INP must pass causal large-network preflight V3")
    no_control = preflight.get("no_control_contract")
    if not isinstance(no_control, dict) or no_control.get("id") != "NO_SUPERVISORY_RTC_V2":
        raise ValueError("INP preflight lacks the No-supervisory-RTC V2 contract")

    readiness = legacy._json(artefacts["study_readiness"])
    if readiness.get("contract") != READINESS_CONTRACT or readiness.get("passed") is not True:
        raise ValueError("Policy Lock requires a passed WUHAN_RTC_PRETRAINING_READINESS_V1 artifact")
    if str(readiness.get("event_registry_sha256", "")) != sha256_file(artefacts["split_registry"]):
        raise ValueError("study readiness belongs to a different prepared event registry")
    if str(readiness.get("frozen_inp_sha256", "")) != sha256_file(artefacts["frozen_inp"]):
        raise ValueError("study readiness belongs to a different frozen INP")
    if str(readiness.get("sensor_layout_sha256", "")) != sha256_file(artefacts["sensor_layout"]):
        raise ValueError("study readiness belongs to a different sensor layout")
    actuation_scope = str(readiness.get("actuation_scope", ""))
    field_claim = bool(readiness.get("field_deployment_claim", False))
    if actuation_scope == "SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY" and field_claim:
        raise ValueError("simulation-only Policy Lock cannot make a field-deployment actuation claim")

    priority = legacy._lines(artefacts["priority_nodes"])
    sensors = legacy._lines(artefacts["sensor_layout"])
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("Wuhan reporting requires exactly eight verified priority nodes")
    if not sensors:
        raise ValueError("sensor layout cannot be empty")
    with np.load(artefacts["graph_schema"], allow_pickle=False) as graph:
        node_ids = set(graph["node_ids"].astype(str).tolist())
    missing_nodes = sorted((set(priority) | set(sensors)) - node_ids)
    if missing_nodes:
        raise ValueError(f"priority/sensor nodes are absent from graph: {missing_nodes}")

    controller = legacy._json(artefacts["controller_config"])
    if controller.get("contract") != PRODUCTION_CONTROLLER_CONTRACT:
        raise ValueError("Policy Lock requires production controller config V5 temporal continuity")
    project7_runtime = validate_project7_runtime_config(controller)
    if controller.get("exact_global_peak") is not False:
        raise ValueError("main control runs must keep exact_global_peak=false")
    if "flood_budget_m3" in controller or "depth_budget_m" in controller:
        raise ValueError("PFV/depth hard budgets are incompatible with TFV-first control")
    objective = controller.get("objective")
    if not isinstance(objective, dict) or objective.get("priority_role") != (
        "soft lexicographic secondary preference within TFV near-optimal set"
    ):
        raise ValueError("controller does not declare the frozen TFV-first objective")

    timing = project7_runtime["timing"]
    assert isinstance(timing, dict)
    timing_obj = legacy.timing_from_controller_config(controller)
    if float(readiness.get("minimum_pre_rain_warmup_minutes", -1.0)) < 120.0:
        raise ValueError("prepared events do not satisfy the effective 120-minute warm-up contract")
    controller_section = controller.get("controller")
    if not isinstance(controller_section, dict):
        raise ValueError("controller config lacks controller section")
    time_cfg = legacy._json(artefacts["time_scale_config"])
    for key in ("model_step_seconds", "control_update_seconds"):
        if int(controller[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")
    for key in ("history_steps", "horizon_steps"):
        if key in time_cfg and int(controller_section[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")

    model_contracts = legacy._verify_model_timing(
        artefacts,
        implementation_sha=implementation_sha,
        model_step_seconds=timing_obj.model_step_seconds,
        history_steps=int(controller_section["history_steps"]),
        horizon_steps=int(controller_section["horizon_steps"]),
    )
    legacy._verify_acceptance(artefacts, implementation_sha)
    runtime_acceptance = _verify_runtime_acceptance_v2(
        artefacts["runtime_acceptance"],
        controller_config_path=artefacts["controller_config"],
        control_update_seconds=timing_obj.control_update_seconds,
        implementation_sha=implementation_sha,
    )

    plan = legacy._json(artefacts["baseline_plan"])
    if plan.get("contract") != "FORMAL_BASELINE_PLAN_V6_EVENT_PAIRED_INFORMATION_DISCLOSED":
        raise ValueError("Policy Lock requires Formal baseline plan V6 event-paired semantics")
    strategies = tuple(str(x) for x in plan.get("strategies", []))
    if strategies != EXPECTED_STRATEGIES:
        raise ValueError(f"Formal strategy matrix must be exactly {list(EXPECTED_STRATEGIES)}")

    hashes = {name: sha256_file(path) for name, path in sorted(artefacts.items())}
    canonical = json.dumps(
        {"implementation_contract_sha256": implementation_sha, "artefacts": hashes},
        sort_keys=True,
        separators=(",", ":"),
    )
    policy_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "contract": POLICY_LOCK_CONTRACT,
        "policy_sha256": policy_sha,
        "rtc_source_tree_sha256": implementation_sha,
        "implementation_binding": "semantic_scientific_contract_plus_exact_numerical_artifact_hashes",
        "physical_network_sha256": physical_contract_sha256(artefacts["frozen_inp"]),
        "native_controls_payload_sha256": section_payload_sha256(artefacts["frozen_inp"], "CONTROLS"),
        "objective_contract": "TFV_PRIMARY__PRIORITY_PFV_SOFT_SECONDARY_V1",
        "priority_is_hard_constraint": False,
        "priority_nodes": list(priority),
        "sensor_nodes": list(sensors),
        "causal_timing": timing_obj.as_dict(),
        "project7_runtime_contract": project7_runtime,
        "model_contracts": model_contracts,
        "rainfall_design": rainfall_design,
        "rainfall_sample_size_is_execution_gate": False,
        "study_readiness": readiness,
        "actuation_scope": actuation_scope,
        "field_deployment_claim": field_claim,
        "no_control_contract": no_control,
        "runtime_acceptance": runtime_acceptance,
        "formal_strategy_matrix": list(strategies),
        "competitive_baselines": ["no_control", "internal_rtc", "auto_rbc", "efd"],
        "diagnostic_extremes": ["all_open", "all_closed"],
        "baseline_information_budget": "Internal RTC/Auto-RBC/EFD true-state advantage explicitly accepted and disclosed",
        "formal_metric_aggregation": "equal_weight_per_independent_rainfall_group",
        "workspace": workspace,
        "artifact_location_is_execution_gate": False,
        "artefacts": artefacts,
        "sha256": hashes,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create seven-strategy TFV Policy Lock")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(create_policy_lock(
        ledger_path=args.ledger,
        artefacts_path=args.artifacts,
        output_path=args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
