from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .causal_timing import timing_from_controller_config
from .code_contract import rtc_source_tree_sha256
from .fresh_workspace import load_fresh_workspace
from .inp_lineage import physical_contract_sha256
from .rainfall_design import validate_formal_rainfall_design
from .tfv_pipeline import TFVPipelineLedger, sha256_file


POLICY_LOCK_CONTRACT = "WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND"

_REQUIRED = {
    "fresh_workspace_manifest",
    "inp_preflight",
    "frozen_inp",
    "priority_nodes",
    "sensor_layout",
    "time_scale_config",
    "step1_model",
    "step2_model",
    "graph_schema",
    "split_registry",
    "model_acceptance_contract",
    "step1_acceptance",
    "step2_acceptance",
    "gradient_acceptance",
    "candidate_ranking_acceptance",
    "controller_config",
    "baseline_plan",
    "runtime_acceptance",
}

_EXPECTED_GATE_SOURCE_CONTRACTS = {
    "step1_acceptance": "STEP1_HELDOUT_ACCEPTANCE_V4_GROUP_BALANCED_T0_ENGINE_BOUND",
    "step2_acceptance": "STEP2_EXACT_TRUTH_ACCEPTANCE_V4_GROUP_BALANCED_TIME_LOCKED",
    "gradient_acceptance": "D2_SWMM_TFV_GRADIENT_METRICS_V4_BOUND_AWARE_GROUP_BALANCED",
}
_EXPECTED_GATE_AGGREGATION = {
    "step1_acceptance": "equal_weight_per_rainfall_group",
    "step2_acceptance": "equal_weight_per_rainfall_group",
    "gradient_acceptance": "equal_weight_per_rainfall_group",
}
RANKING_CONTRACT = "SWMM_JOINT_ACTION_RANKING_ACCEPTANCE_V5_D2_D3_GROUP_BALANCED"


def _json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _thresholds(value: object) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("acceptance threshold section must be an object")
    minimum = {str(k): float(v) for k, v in value.get("minimum", {}).items()}
    maximum = {str(k): float(v) for k, v in value.get("maximum", {}).items()}
    if not minimum and not maximum:
        raise ValueError("acceptance threshold section is empty")
    return {"minimum": minimum, "maximum": maximum}


def _verify_workspace_and_split(
    artefacts: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    workspace = load_fresh_workspace(artefacts["fresh_workspace_manifest"])
    inputs = workspace.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("event_registry"), dict):
        raise ValueError("workspace lacks the bound event-registry identity")
    event_identity = inputs["event_registry"]
    assert isinstance(event_identity, dict)
    if sha256_file(artefacts["split_registry"]) != str(event_identity.get("sha256", "")):
        raise ValueError(
            "Policy Lock split_registry differs from the canonical registry bound to the workspace"
        )
    import pandas as pd

    design = validate_formal_rainfall_design(pd.read_csv(artefacts["split_registry"]))
    return workspace, design


def _verify_gate_source_file(evidence: dict[str, object]) -> None:
    source_path = Path(str(evidence.get("source_metrics_path", "")))
    source_sha = str(evidence.get("source_metrics_sha256", ""))
    if not source_path.is_file() or not source_sha or sha256_file(source_path) != source_sha:
        raise ValueError(f"acceptance source metrics disappeared/changed: {source_path}")


def _verify_acceptance(artefacts: dict[str, str], implementation_sha: str) -> None:
    contract = _json(artefacts["model_acceptance_contract"])
    if contract.get("contract") != "MODEL_ACCEPTANCE_CONTRACT_V3_TFV_FIRST":
        raise ValueError("Policy Lock requires MODEL_ACCEPTANCE_CONTRACT_V3_TFV_FIRST")
    step1_sha = sha256_file(artefacts["step1_model"])
    step2_sha = sha256_file(artefacts["step2_model"])

    for name, section in (
        ("step1_acceptance", "step1"),
        ("step2_acceptance", "step2"),
        ("gradient_acceptance", "gradient"),
    ):
        gate = _json(artefacts[name])
        if gate.get("contract") != "PREREGISTERED_ACCEPTANCE_GATE_V4_SOURCE_BOUND":
            raise ValueError(f"{name} is not a preregistered source-bound acceptance gate")
        if gate.get("section") != section or gate.get("passed") is not True:
            raise ValueError(f"{name} did not pass the expected section {section}")
        if gate.get("rtc_source_tree_sha256") != implementation_sha:
            raise ValueError(f"{name} uses an incompatible implementation contract")
        if gate.get("source_metrics_contract") != _EXPECTED_GATE_SOURCE_CONTRACTS[name]:
            raise ValueError(f"{name} was computed from an incompatible metric contract")
        if gate.get("source_metrics_aggregation") != _EXPECTED_GATE_AGGREGATION[name]:
            raise ValueError(f"{name} is not rainfall-group balanced")
        _verify_gate_source_file(gate)
        if _thresholds(contract.get(section)) != _thresholds(gate.get("thresholds", {})):
            raise ValueError(f"{name} thresholds differ from the frozen acceptance contract")

    if str(_json(artefacts["step1_acceptance"]).get("model_sha256", "")) != step1_sha:
        raise ValueError("Step1 acceptance does not belong to the locked Step1 model")
    if str(_json(artefacts["step2_acceptance"]).get("model_sha256", "")) != step2_sha:
        raise ValueError("Step2 acceptance does not belong to the locked Step2 model")
    if str(_json(artefacts["gradient_acceptance"]).get("step2_sha256", "")) != step2_sha:
        raise ValueError("gradient acceptance does not belong to the locked Step2 model")

    ranking = _json(artefacts["candidate_ranking_acceptance"])
    if ranking.get("contract") != RANKING_CONTRACT or ranking.get("passed") is not True:
        raise ValueError("D2 local + D3 joint candidate ranking acceptance must pass")
    if ranking.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError("candidate ranking uses an incompatible implementation contract")
    if ranking.get("aggregation") != "checkpoint_metrics_then_equal_weight_per_rainfall_group":
        raise ValueError("candidate ranking is not rainfall-group balanced")
    if str(ranking.get("step2_sha256", "")) != step2_sha:
        raise ValueError("candidate ranking does not belong to the locked Step2 model")
    if _thresholds(contract.get("candidate_ranking")) != _thresholds(
        ranking.get("thresholds", {})
    ):
        raise ValueError("candidate-ranking thresholds differ from the frozen contract")


def _load_checkpoint(
    path: str | Path, *, name: str, implementation_sha: str
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{name} checkpoint payload is invalid")
    if payload.get("checkpoint_contract") != "RTC_TORCH_CHECKPOINT_V2_CODE_BOUND":
        raise ValueError(f"{name} checkpoint uses an unsupported checkpoint contract")
    if payload.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError(f"{name} checkpoint uses an incompatible implementation contract")
    if payload.get("scientific_split") != "development":
        raise ValueError(f"{name} must be fitted on development data only")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"{name} checkpoint lacks model_config")
    if not payload.get("training_manifest_sha256"):
        raise ValueError(f"{name} checkpoint lacks training-data manifest lineage")
    return payload


def _verify_model_timing(
    artefacts: dict[str, str],
    *,
    implementation_sha: str,
    model_step_seconds: int,
    history_steps: int,
    horizon_steps: int,
) -> dict[str, object]:
    step1 = _load_checkpoint(
        artefacts["step1_model"], name="Step1", implementation_sha=implementation_sha
    )
    step2 = _load_checkpoint(
        artefacts["step2_model"], name="Step2", implementation_sha=implementation_sha
    )
    step1_cfg = dict(step1["model_config"])
    step2_cfg = dict(step2["model_config"])
    if int(step1_cfg.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError("Step1 model step differs from controller")
    if int(step1_cfg.get("history_steps", -1)) != history_steps:
        raise ValueError("Step1 history length differs from controller")
    if int(step2_cfg.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError("Step2 model step differs from controller")
    if int(step2_cfg.get("horizon_steps", -1)) != horizon_steps:
        raise ValueError("Step2 horizon differs from controller")
    if step2_cfg.get("time_contract") != "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2":
        raise ValueError("Step2 checkpoint lacks the frozen discrete-time/engine contract")
    step1_engine = str(step1_cfg.get("swmm_engine_version", "")).strip()
    step2_engine = str(step2_cfg.get("swmm_engine_version", "")).strip()
    if not step1_engine or step1_engine != step2_engine:
        raise ValueError(
            f"Step1/Step2 SWMM engine lineage differs: {step1_engine} != {step2_engine}"
        )
    return {
        "step1_training_manifest_sha256": str(step1["training_manifest_sha256"]),
        "step2_training_manifest_sha256": str(step2["training_manifest_sha256"]),
        "model_step_seconds": model_step_seconds,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "swmm_engine_version": step1_engine,
    }


def _verify_runtime_acceptance(
    path: str | Path,
    *,
    controller_config_path: str | Path,
    control_update_seconds: int,
    implementation_sha: str,
) -> dict[str, object]:
    evidence = _json(path)
    if evidence.get("contract") != "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V1":
        raise ValueError("Policy Lock requires development real-time execution acceptance")
    if evidence.get("passed") is not True:
        raise ValueError("development real-time execution acceptance did not pass")
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
    for key in (
        "control_grid_violations",
        "first_decision_violations",
        "missing_runtime_diagnostics",
        "fatal_runtime_fallbacks",
    ):
        if int(metrics.get(key, -1)) != 0:
            raise ValueError(f"runtime acceptance contains nonzero {key}")
    if float(metrics.get("decision_runtime_max_seconds", float("inf"))) > budget:
        raise ValueError("development decision runtime exceeded the locked budget")
    return evidence


def create_policy_lock(
    *,
    ledger_path: str | Path,
    artefacts_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    ledger = TFVPipelineLedger.from_json(ledger_path)
    ledger.require_ready_for_lock()

    raw = _json(artefacts_path)
    artefacts = {str(k): str(v) for k, v in raw.items()}
    missing = sorted(_REQUIRED - set(artefacts))
    if missing:
        raise ValueError(f"Policy Lock missing required artifacts: {missing}")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"Policy Lock artifact missing: {name}: {path}")

    implementation_sha = rtc_source_tree_sha256()
    workspace, rainfall_design = _verify_workspace_and_split(artefacts)

    preflight = _json(artefacts["inp_preflight"])
    if preflight.get("contract") != "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC":
        raise ValueError("frozen INP must pass causal large-network preflight V3")
    no_control = preflight.get("no_control_contract")
    if not isinstance(no_control, dict) or no_control.get("id") != "NO_SUPERVISORY_RTC_V2":
        raise ValueError("INP preflight lacks the No-supervisory-RTC V2 contract")

    priority = _lines(artefacts["priority_nodes"])
    sensors = _lines(artefacts["sensor_layout"])
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("Wuhan reporting requires exactly eight verified priority nodes")
    if not sensors:
        raise ValueError("sensor layout cannot be empty")
    with np.load(artefacts["graph_schema"], allow_pickle=False) as graph:
        node_ids = set(graph["node_ids"].astype(str).tolist())
    missing_nodes = sorted((set(priority) | set(sensors)) - node_ids)
    if missing_nodes:
        raise ValueError(f"priority/sensor nodes are absent from graph: {missing_nodes}")

    controller = _json(artefacts["controller_config"])
    if controller.get("contract") != "PRODUCTION_CONTROLLER_CONFIG_V4_TFV_FIRST":
        raise ValueError("Policy Lock requires production TFV-first controller config V4")
    if controller.get("exact_global_peak") is not False:
        raise ValueError("main control runs must keep exact_global_peak=false and replay it later")
    if "flood_budget_m3" in controller or "depth_budget_m" in controller:
        raise ValueError("PFV/depth hard budgets are incompatible with the TFV-first objective")
    objective = controller.get("objective")
    if not isinstance(objective, dict) or objective.get("priority_role") != (
        "soft lexicographic secondary preference within TFV near-optimal set"
    ):
        raise ValueError("controller does not declare the frozen TFV-first soft-priority objective")

    timing = timing_from_controller_config(controller)
    timing.validate(require_full_history_before_first_control=True)
    controller_section = controller.get("controller")
    if not isinstance(controller_section, dict):
        raise ValueError("controller config lacks controller section")
    time_cfg = _json(artefacts["time_scale_config"])
    for key in ("model_step_seconds", "control_update_seconds"):
        if int(controller[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")
    for key in ("history_steps", "horizon_steps"):
        if key in time_cfg and int(controller_section[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")

    model_contracts = _verify_model_timing(
        artefacts,
        implementation_sha=implementation_sha,
        model_step_seconds=timing.model_step_seconds,
        history_steps=int(controller_section["history_steps"]),
        horizon_steps=int(controller_section["horizon_steps"]),
    )
    _verify_acceptance(artefacts, implementation_sha)
    runtime_acceptance = _verify_runtime_acceptance(
        artefacts["runtime_acceptance"],
        controller_config_path=artefacts["controller_config"],
        control_update_seconds=timing.control_update_seconds,
        implementation_sha=implementation_sha,
    )

    plan = _json(artefacts["baseline_plan"])
    if plan.get("contract") != "FORMAL_BASELINE_PLAN_V4_NO_DUPLICATE_HOLD":
        raise ValueError("Policy Lock requires Formal baseline plan V4")
    strategies = tuple(str(x) for x in plan.get("strategies", []))
    expected_strategies = (
        "proposed",
        "no_control",
        "internal_rtc",
        "all_open",
        "all_closed",
    )
    if strategies != expected_strategies:
        raise ValueError(
            "Formal strategy matrix must be exactly proposed/no_control/internal_rtc/all_open/all_closed"
        )

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
        "objective_contract": "TFV_PRIMARY__PRIORITY_PFV_SOFT_SECONDARY_V1",
        "priority_is_hard_constraint": False,
        "priority_nodes": list(priority),
        "sensor_nodes": list(sensors),
        "causal_timing": timing.as_dict(),
        "model_contracts": model_contracts,
        "rainfall_design": rainfall_design,
        "rainfall_sample_size_is_execution_gate": False,
        "no_control_contract": no_control,
        "runtime_acceptance": runtime_acceptance,
        "formal_strategy_matrix": list(strategies),
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
    parser = argparse.ArgumentParser(
        description="Create the science-first TFV Policy Lock with time/data/engine binding"
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = create_policy_lock(
        ledger_path=args.ledger,
        artefacts_path=args.artifacts,
        output_path=args.out,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
