from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import policy_lock as legacy
from .code_contract import rtc_source_tree_sha256
from .control_lineage import section_payload_sha256
from .inp_lineage import physical_contract_sha256
from .project7_contract import PRODUCTION_CONTROLLER_CONTRACT, validate_project7_runtime_config
from .study_readiness import READINESS_CONTRACT
from .tfv_pipeline import TFVPipelineLedger, sha256_file

POLICY_LOCK_CONTRACT = legacy.POLICY_LOCK_CONTRACT
SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
ACCEPTANCE_CONTRACT = "MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED"
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
    if not np.isclose(budget, 300.0, rtol=0.0, atol=1e-9):
        raise ValueError("v0.6.9 Policy Lock requires the frozen 300 s decision runtime budget")
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


def _verify_acceptance_v4(artefacts: dict[str, str], implementation_sha: str) -> None:
    contract = legacy._json(artefacts["model_acceptance_contract"])
    if contract.get("contract") != ACCEPTANCE_CONTRACT:
        raise ValueError(f"Policy Lock requires {ACCEPTANCE_CONTRACT}")
    step1_sha = sha256_file(artefacts["step1_model"])
    step2_sha = sha256_file(artefacts["step2_model"])

    for name, section in (
        ("step1_acceptance", "step1"),
        ("step2_acceptance", "step2"),
        ("gradient_acceptance", "gradient"),
    ):
        gate = legacy._json(artefacts[name])
        if gate.get("contract") != "PREREGISTERED_ACCEPTANCE_GATE_V4_SOURCE_BOUND":
            raise ValueError(f"{name} is not a preregistered source-bound acceptance gate")
        if gate.get("section") != section or gate.get("passed") is not True:
            raise ValueError(f"{name} did not pass the expected section {section}")
        if gate.get("rtc_source_tree_sha256") != implementation_sha:
            raise ValueError(f"{name} uses an incompatible implementation contract")
        if gate.get("source_metrics_contract") != legacy._EXPECTED_GATE_SOURCE_CONTRACTS[name]:
            raise ValueError(f"{name} was computed from an incompatible metric contract")
        if gate.get("source_metrics_aggregation") != legacy._EXPECTED_GATE_AGGREGATION[name]:
            raise ValueError(f"{name} is not rainfall-group balanced")
        legacy._verify_gate_source_file(gate)
        if legacy._thresholds(contract.get(section)) != legacy._thresholds(
            gate.get("thresholds", {})
        ):
            raise ValueError(f"{name} thresholds differ from the frozen acceptance contract")

    if str(legacy._json(artefacts["step1_acceptance"]).get("model_sha256", "")) != step1_sha:
        raise ValueError("Step1 acceptance does not belong to the locked Step1 model")
    if str(legacy._json(artefacts["step2_acceptance"]).get("model_sha256", "")) != step2_sha:
        raise ValueError("Step2 acceptance does not belong to the locked Step2 model")
    if str(legacy._json(artefacts["gradient_acceptance"]).get("step2_sha256", "")) != step2_sha:
        raise ValueError("gradient acceptance does not belong to the locked Step2 model")

    ranking = legacy._json(artefacts["candidate_ranking_acceptance"])
    if ranking.get("contract") != legacy.RANKING_CONTRACT or ranking.get("passed") is not True:
        raise ValueError("D2 local + D3 joint candidate ranking acceptance must pass")
    if ranking.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError("candidate ranking uses an incompatible implementation contract")
    if ranking.get("aggregation") != "checkpoint_metrics_then_equal_weight_per_rainfall_group":
        raise ValueError("candidate ranking is not rainfall-group balanced")
    if str(ranking.get("step2_sha256", "")) != step2_sha:
        raise ValueError("candidate ranking does not belong to the locked Step2 model")
    if legacy._thresholds(contract.get("candidate_ranking")) != legacy._thresholds(
        ranking.get("thresholds", {})
    ):
        raise ValueError("candidate-ranking thresholds differ from the frozen contract")


def _verify_split_contract(
    artefacts: dict[str, str], rainfall_design: dict[str, object]
) -> dict[str, object]:
    payload = legacy._json(artefacts["split_contract"])
    if payload.get("contract") != SPLIT_CONTRACT:
        raise ValueError(f"Policy Lock requires split contract {SPLIT_CONTRACT}")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or {
        "development_train": int(counts.get("development_train", -1)),
        "development_validation": int(counts.get("development_validation", -1)),
        "final": int(counts.get("final", -1)),
    } != {"development_train": 18, "development_validation": 6, "final": 6}:
        raise ValueError("Policy Lock split contract is not the frozen 18/6/6 allocation")

    registry = pd.read_csv(artefacts["split_registry"], keep_default_na=False)
    train = set(
        registry.loc[
            (registry["scientific_split"].astype(str) == "development")
            & (registry["development_fold"].astype(str) == "train"),
            "event_id",
        ].astype(str)
    )
    validation = set(
        registry.loc[
            (registry["scientific_split"].astype(str) == "development")
            & (registry["development_fold"].astype(str) == "validation"),
            "event_id",
        ].astype(str)
    )
    final = set(
        registry.loc[registry["scientific_split"].astype(str) == "final", "event_id"].astype(str)
    )
    if train != set(str(x) for x in payload.get("development_train", [])):
        raise ValueError("prepared Train event IDs differ from the frozen split contract")
    if validation != set(str(x) for x in payload.get("development_validation", [])):
        raise ValueError("prepared Validation event IDs differ from the frozen split contract")
    if final != set(str(x) for x in payload.get("final", [])):
        raise ValueError("prepared Final event IDs differ from the frozen split contract")
    if int(rainfall_design.get("development_train_groups", -1)) != 18:
        raise ValueError("rainfall-design evidence does not prove 18 Train groups")
    if int(rainfall_design.get("development_validation_groups", -1)) != 6:
        raise ValueError("rainfall-design evidence does not prove 6 Validation groups")
    if int(rainfall_design.get("final_groups", -1)) != 6:
        raise ValueError("rainfall-design evidence does not prove 6 Final groups")
    return payload


def create_policy_lock(
    *, ledger_path: str | Path, artefacts_path: str | Path, output_path: str | Path
) -> dict[str, object]:
    ledger = TFVPipelineLedger.from_json(ledger_path)
    ledger.require_ready_for_lock()
    raw = legacy._json(artefacts_path)
    artefacts = {str(k): str(v) for k, v in raw.items()}
    required = set(legacy._REQUIRED) | {"study_readiness", "split_contract"}
    missing = sorted(required - set(artefacts))
    if missing:
        raise ValueError(f"Policy Lock missing required artifacts: {missing}")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"Policy Lock artifact missing: {name}: {path}")

    implementation_sha = rtc_source_tree_sha256()
    workspace, rainfall_design = legacy._verify_workspace_and_split(artefacts)
    split_contract = _verify_split_contract(artefacts, rainfall_design)
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
    controller_section = controller.get("controller")
    if not isinstance(controller_section, dict):
        raise ValueError("controller config lacks controller section")
    if not np.isclose(float(controller_section.get("readback_target_tolerance", -1)), 1e-6):
        raise ValueError("Policy Lock requires readback_target_tolerance=1e-6")
    if not np.isclose(float(controller_section.get("readback_current_tolerance", -1)), 0.05):
        raise ValueError("Policy Lock requires readback_current_tolerance=0.05")
    if not np.isclose(float(controller_section.get("decision_runtime_budget_seconds", -1)), 300.0):
        raise ValueError("Policy Lock requires decision_runtime_budget_seconds=300")

    timing = project7_runtime["timing"]
    assert isinstance(timing, dict)
    timing_obj = legacy.timing_from_controller_config(controller)
    if float(readiness.get("minimum_pre_rain_warmup_minutes", -1.0)) < 120.0:
        raise ValueError("prepared events do not satisfy the effective 120-minute warm-up contract")
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
    _verify_acceptance_v4(artefacts, implementation_sha)
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
        "split_contract": split_contract,
        "split_contract_sha256": sha256_file(artefacts["split_contract"]),
        "rainfall_sample_size_is_execution_gate": True,
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
    parser = argparse.ArgumentParser(description="Create v0.6.9 execution-frozen seven-strategy TFV Policy Lock")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_policy_lock(
                ledger_path=args.ledger,
                artefacts_path=args.artifacts,
                output_path=args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
