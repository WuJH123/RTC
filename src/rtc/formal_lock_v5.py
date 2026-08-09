from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .causal_timing import timing_from_controller_config
from .code_contract import rtc_source_tree_sha256
from .fresh_workspace import load_fresh_workspace, require_path_inside_workspace
from .inp_lineage import physical_contract_sha256
from .tfv_pipeline import TFVPipelineLedger, sha256_file


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
    "state_schema",
    "actuator_catalog",
    "split_registry",
    "model_acceptance_contract",
    "step1_acceptance",
    "step2_acceptance",
    "gradient_acceptance",
    "candidate_ranking_acceptance",
    "controller_config",
    "rainfall_forecast_config",
    "fallback_policy",
    "baseline_plan",
    "runtime_acceptance",
}

_WORKSPACE_OUTPUT_ARTIFACTS = {
    "inp_preflight",
    "time_scale_config",
    "step1_model",
    "step2_model",
    "graph_schema",
    "state_schema",
    "actuator_catalog",
    "split_registry",
    "step1_acceptance",
    "step2_acceptance",
    "gradient_acceptance",
    "candidate_ranking_acceptance",
    "controller_config",
    "rainfall_forecast_config",
    "runtime_acceptance",
}

FORMAL_MIN_RAINFALL_GROUPS = 160
FORMAL_MIN_ROLE_GROUPS = {
    "development": 96,
    "calibration": 24,
    "safety_audit": 16,
    "final": 24,
}
FORMAL_MIN_DEV_VALIDATION_GROUPS = 19

_EXPECTED_GATE_SOURCE_CONTRACTS = {
    "step1_acceptance": "STEP1_HELDOUT_ACCEPTANCE_V3_GROUP_BALANCED_TIME_LOCKED",
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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _thresholds(value: object) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("acceptance threshold section must be an object")
    result = {
        "minimum": {
            str(k): float(v) for k, v in value.get("minimum", {}).items()
        },
        "maximum": {
            str(k): float(v) for k, v in value.get("maximum", {}).items()
        },
    }
    if not result["minimum"] and not result["maximum"]:
        raise ValueError("formal threshold section is empty")
    return result


def _verify_splits(path: str | Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    required_columns = {"rainfall_group", "scientific_split", "development_fold"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"split registry lacks columns: {missing_columns}")
    frame = frame.copy()
    frame["rainfall_group"] = frame["rainfall_group"].astype(str)
    frame["scientific_split"] = frame["scientific_split"].astype(str)
    cross = frame.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("rainfall-group leakage exists across scientific splits")
    total_groups = int(frame["rainfall_group"].nunique())
    if total_groups < FORMAL_MIN_RAINFALL_GROUPS:
        raise ValueError(
            f"Formal fresh-data design requires at least {FORMAL_MIN_RAINFALL_GROUPS} "
            f"independent rainfall groups; got {total_groups}"
        )
    role_counts = (
        frame[["rainfall_group", "scientific_split"]]
        .drop_duplicates()
        .groupby("scientific_split")["rainfall_group"]
        .count()
        .to_dict()
    )
    for role, minimum in FORMAL_MIN_ROLE_GROUPS.items():
        if int(role_counts.get(role, 0)) < minimum:
            raise ValueError(
                f"Formal rainfall split needs at least {minimum} {role} groups; "
                f"got {role_counts.get(role, 0)}"
            )
    dev = frame[frame["scientific_split"] == "development"]
    if set(dev["development_fold"].astype(str)) != {"train", "validation"}:
        raise ValueError(
            "development split must contain rainfall-group-disjoint train/validation folds"
        )
    if (dev.groupby("rainfall_group")["development_fold"].nunique() != 1).any():
        raise ValueError("rainfall group crosses development train/validation folds")
    validation_groups = int(
        dev.loc[
            dev["development_fold"].astype(str) == "validation", "rainfall_group"
        ].nunique()
    )
    if validation_groups < FORMAL_MIN_DEV_VALIDATION_GROUPS:
        raise ValueError(
            f"development validation requires at least {FORMAL_MIN_DEV_VALIDATION_GROUPS} "
            f"rainfall groups; got {validation_groups}"
        )
    non_dev = frame[frame["scientific_split"] != "development"]
    if (non_dev["development_fold"].fillna("").astype(str) != "").any():
        raise ValueError("non-development rainfall groups must not carry a development fold")
    return {
        "contract": "FORMAL_FRESH_RAINFALL_COHORTS_V1",
        "minimum_total_groups": FORMAL_MIN_RAINFALL_GROUPS,
        "total_groups": total_groups,
        "role_group_counts": {k: int(v) for k, v in role_counts.items()},
        "development_validation_groups": validation_groups,
    }


def _verify_gate_source_file(evidence: dict[str, object]) -> None:
    raw_path = evidence.get("source_metrics_path")
    raw_sha = evidence.get("source_metrics_sha256")
    if not raw_path or not raw_sha:
        raise ValueError("acceptance gate lacks source metric path/hash")
    source = Path(str(raw_path))
    if not source.is_file() or sha256_file(source) != str(raw_sha):
        raise ValueError(f"acceptance source metrics disappeared/changed: {source}")


def _verify_acceptance(artefacts: dict[str, str], current_code_sha: str) -> None:
    contract = _json(artefacts["model_acceptance_contract"])
    if contract.get("contract") != "MODEL_ACCEPTANCE_CONTRACT_V3_TFV_FIRST":
        raise ValueError(
            "TFV-first lock requires MODEL_ACCEPTANCE_CONTRACT_V3_TFV_FIRST"
        )
    expected_step1 = sha256_file(artefacts["step1_model"])
    expected_step2 = sha256_file(artefacts["step2_model"])

    for name, section in (
        ("step1_acceptance", "step1"),
        ("step2_acceptance", "step2"),
        ("gradient_acceptance", "gradient"),
    ):
        evidence = _json(artefacts[name])
        if evidence.get("contract") != "PREREGISTERED_ACCEPTANCE_GATE_V4_SOURCE_BOUND":
            raise ValueError(f"{name} is not a current source-bound preregistered gate")
        if evidence.get("section") != section or evidence.get("passed") is not True:
            raise ValueError(f"formal evidence did not pass correct section: {name}")
        if evidence.get("rtc_source_tree_sha256") != current_code_sha:
            raise ValueError(f"{name} was generated by a different RTC source tree")
        if evidence.get("source_metrics_contract") != _EXPECTED_GATE_SOURCE_CONTRACTS[name]:
            raise ValueError(
                f"{name} source metric contract is stale/wrong: "
                f"{evidence.get('source_metrics_contract')}"
            )
        if evidence.get("source_metrics_aggregation") != _EXPECTED_GATE_AGGREGATION[name]:
            raise ValueError(f"{name} is not rainfall-group balanced")
        _verify_gate_source_file(evidence)
        frozen = _thresholds(contract.get(section))
        used = _thresholds(evidence.get("thresholds", {}))
        if frozen != used:
            raise ValueError(f"{name} thresholds differ from preregistered contract")
        metrics = evidence.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{name} lacks metrics")
        missing = (set(frozen["minimum"]) | set(frozen["maximum"])) - set(metrics)
        if missing:
            raise ValueError(
                f"{name} missing thresholded metrics: {sorted(missing)}"
            )

    step1 = _json(artefacts["step1_acceptance"])
    step2 = _json(artefacts["step2_acceptance"])
    gradient = _json(artefacts["gradient_acceptance"])
    if str(step1.get("model_sha256", "")) != expected_step1:
        raise ValueError("Step1 acceptance was not generated by locked Step1 model")
    if str(step2.get("model_sha256", "")) != expected_step2:
        raise ValueError("Step2 acceptance was not generated by locked Step2 model")
    if str(gradient.get("step2_sha256", "")) != expected_step2:
        raise ValueError("gradient acceptance was not generated by locked Step2 model")

    ranking = _json(artefacts["candidate_ranking_acceptance"])
    if ranking.get("contract") != RANKING_CONTRACT or ranking.get("passed") is not True:
        raise ValueError("candidate ranking must pass current D2+D3 group-balanced gate")
    if ranking.get("rtc_source_tree_sha256") != current_code_sha:
        raise ValueError("candidate ranking was generated by a different RTC source tree")
    if ranking.get("aggregation") != "checkpoint_metrics_then_equal_weight_per_rainfall_group":
        raise ValueError("candidate ranking is not rainfall-group balanced")
    if str(ranking.get("step2_sha256", "")) != expected_step2:
        raise ValueError("candidate ranking was not generated by locked Step2 model")
    frozen_ranking = _thresholds(contract.get("candidate_ranking"))
    used_ranking = _thresholds(ranking.get("thresholds", {}))
    if frozen_ranking != used_ranking:
        raise ValueError("candidate ranking thresholds differ from preregistered contract")
    ranking_metrics = ranking.get("metrics")
    if not isinstance(ranking_metrics, dict):
        raise ValueError("candidate ranking lacks metrics")
    missing_ranking = (
        set(frozen_ranking["minimum"]) | set(frozen_ranking["maximum"])
    ) - set(ranking_metrics)
    if missing_ranking:
        raise ValueError(
            f"candidate ranking missing thresholded metrics: {sorted(missing_ranking)}"
        )


def _load_current_checkpoint(path: str | Path, *, name: str, current_code_sha: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{name} checkpoint payload is invalid")
    if payload.get("checkpoint_contract") != "RTC_TORCH_CHECKPOINT_V2_CODE_BOUND":
        raise ValueError(f"{name} checkpoint is not code-bound V2")
    if payload.get("rtc_source_tree_sha256") != current_code_sha:
        raise ValueError(f"{name} checkpoint was trained by a different RTC source tree")
    if payload.get("scientific_split") != "development":
        raise ValueError(f"{name} checkpoint was not trained on development-only data")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"{name} checkpoint lacks model_config")
    if not payload.get("training_manifest_sha256"):
        raise ValueError(f"{name} checkpoint lacks training manifest lineage")
    return payload


def _verify_model_time_contracts(
    artefacts: dict[str, str],
    *,
    current_code_sha: str,
    model_step_seconds: int,
    history_steps: int,
    horizon_steps: int,
) -> dict[str, object]:
    step1 = _load_current_checkpoint(
        artefacts["step1_model"], name="Step1", current_code_sha=current_code_sha
    )
    step2 = _load_current_checkpoint(
        artefacts["step2_model"], name="Step2", current_code_sha=current_code_sha
    )
    step1_cfg = dict(step1["model_config"])
    step2_cfg = dict(step2["model_config"])
    if int(step1_cfg.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError("Step1 checkpoint model step differs from locked controller")
    if int(step1_cfg.get("history_steps", -1)) != history_steps:
        raise ValueError("Step1 checkpoint history differs from locked controller")
    if int(step2_cfg.get("model_step_seconds", -1)) != model_step_seconds:
        raise ValueError("Step2 checkpoint model step differs from locked controller")
    if int(step2_cfg.get("horizon_steps", -1)) != horizon_steps:
        raise ValueError("Step2 checkpoint horizon differs from locked controller")
    if step2_cfg.get("time_contract") != "STEP2_FIXED_DISCRETE_TIME_V1":
        raise ValueError("Step2 checkpoint lacks fixed discrete-time contract")
    if not step1_cfg.get("training_contract_sha256"):
        raise ValueError("Step1 checkpoint lacks resumable training contract lineage")
    if not step2_cfg.get("training_contract_sha256"):
        raise ValueError("Step2 checkpoint lacks resumable training contract lineage")
    return {
        "step1_checkpoint_contract": str(step1["checkpoint_contract"]),
        "step2_checkpoint_contract": str(step2["checkpoint_contract"]),
        "step1_training_manifest_sha256": str(step1["training_manifest_sha256"]),
        "step2_training_manifest_sha256": str(step2["training_manifest_sha256"]),
        "model_step_seconds": model_step_seconds,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
    }


def _verify_runtime_acceptance(
    path: str | Path,
    controller_config_path: str | Path,
    control_update_seconds: int,
    current_code_sha: str,
) -> dict[str, object]:
    evidence = _json(path)
    if evidence.get("contract") != "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V1":
        raise ValueError(
            "Policy Lock requires development real-time execution acceptance V1"
        )
    if evidence.get("passed") is not True:
        raise ValueError("development real-time execution gate did not pass")
    # New evidence records this field; rejecting a missing field intentionally invalidates
    # pre-final audit runs after the source tree has changed.
    if evidence.get("rtc_source_tree_sha256") != current_code_sha:
        raise ValueError("runtime acceptance was generated by a different RTC source tree")
    if str(evidence.get("controller_config_sha256", "")) != sha256_file(
        controller_config_path
    ):
        raise ValueError(
            "runtime acceptance was not generated with the locked controller config"
        )
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


def _verify_fresh_workspace(artefacts: dict[str, str]) -> dict[str, object]:
    workspace = load_fresh_workspace(artefacts["fresh_workspace_manifest"])
    root = Path(str(workspace["output_root"]))
    for name in sorted(_WORKSPACE_OUTPUT_ARTIFACTS):
        require_path_inside_workspace(artefacts[name], root)
    inputs = workspace.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("event_registry"), dict
    ):
        raise ValueError("fresh workspace lacks bound event_registry identity")
    event_identity = inputs["event_registry"]
    assert isinstance(event_identity, dict)
    if sha256_file(artefacts["split_registry"]) != str(
        event_identity.get("sha256", "")
    ):
        raise ValueError(
            "Policy Lock split_registry is not the exact event registry bound to the Fresh Workspace"
        )
    return workspace


def create_formal_policy_lock_v5(
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
        raise ValueError(f"TFV-first Policy Lock missing artefacts: {missing}")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"Policy Lock artifact missing: {name}: {path}")

    current_code_sha = rtc_source_tree_sha256()
    fresh_workspace = _verify_fresh_workspace(artefacts)
    preflight = _json(artefacts["inp_preflight"])
    if preflight.get("contract") != "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC":
        raise ValueError("locked INP must pass causal large-network preflight V3")
    no_control = preflight.get("no_control_contract")
    if (
        not isinstance(no_control, dict)
        or no_control.get("id") != "NO_SUPERVISORY_RTC_V2"
    ):
        raise ValueError("INP preflight lacks the frozen No-supervisory-RTC contract")

    priority = _lines(artefacts["priority_nodes"])
    sensors = _lines(artefacts["sensor_layout"])
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError(
            "Formal Wuhan reporting requires exactly eight verified observed-site node mappings"
        )
    with np.load(artefacts["graph_schema"], allow_pickle=False) as graph:
        node_ids = set(graph["node_ids"].astype(str).tolist())
    missing_nodes = sorted((set(priority) | set(sensors)) - node_ids)
    if missing_nodes:
        raise ValueError(
            f"priority/sensor mapping is incompatible with frozen graph: {missing_nodes}"
        )
    if set(preflight.get("missing_priority_nodes", [])):
        raise ValueError("INP preflight reports unresolved priority nodes")

    rainfall_design = _verify_splits(artefacts["split_registry"])
    controller = _json(artefacts["controller_config"])
    if controller.get("contract") != "PRODUCTION_CONTROLLER_CONFIG_V4_TFV_FIRST":
        raise ValueError("TFV-first lock requires controller config V4")
    if controller.get("exact_global_peak") is not False:
        raise ValueError(
            "Formal main run must use exact_global_peak=false and replay peak afterward"
        )
    if "flood_budget_m3" in controller or "depth_budget_m" in controller:
        raise ValueError(
            "PFV/depth hard budgets are forbidden by the TFV-first scientific contract"
        )
    objective = controller.get("objective")
    if (
        not isinstance(objective, dict)
        or objective.get("priority_role")
        != "soft lexicographic secondary preference within TFV near-optimal set"
    ):
        raise ValueError(
            "controller does not declare the frozen soft-priority TFV-first objective"
        )

    timing = timing_from_controller_config(controller)
    timing.validate(require_full_history_before_first_control=True)
    timing_payload = timing.as_dict()
    controller_section = controller.get("controller")
    if not isinstance(controller_section, dict):
        raise ValueError("controller config lacks controller section")
    raw_budget = controller_section.get("decision_runtime_budget_seconds")
    if raw_budget is None:
        raise ValueError("Formal controller must freeze decision_runtime_budget_seconds")
    decision_budget = float(raw_budget)
    if not 0 < decision_budget < timing.control_update_seconds:
        raise ValueError(
            "decision runtime budget must be positive and less than control interval"
        )

    time_cfg = _json(artefacts["time_scale_config"])
    for key in ("model_step_seconds", "control_update_seconds"):
        if int(controller[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")
    for key in ("history_steps", "horizon_steps"):
        if key in time_cfg and int(controller_section[key]) != int(time_cfg[key]):
            raise ValueError(f"controller/time-scale mismatch: {key}")

    model_contracts = _verify_model_time_contracts(
        artefacts,
        current_code_sha=current_code_sha,
        model_step_seconds=timing.model_step_seconds,
        history_steps=int(controller_section["history_steps"]),
        horizon_steps=int(controller_section["horizon_steps"]),
    )
    _verify_acceptance(artefacts, current_code_sha)
    runtime_acceptance = _verify_runtime_acceptance(
        artefacts["runtime_acceptance"],
        artefacts["controller_config"],
        timing.control_update_seconds,
        current_code_sha,
    )

    plan = _json(artefacts["baseline_plan"])
    if plan.get("contract") != "FORMAL_BASELINE_PLAN_V4_NO_DUPLICATE_HOLD":
        raise ValueError("Formal lock requires the non-duplicate baseline plan V4")
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
            "Formal strategy matrix must be exactly "
            "proposed/no_control/internal_rtc/all_open/all_closed"
        )
    if (
        "hold" in strategies
        or "native_rules" in strategies
        or "passive_no_rtc" in strategies
    ):
        raise ValueError(
            "duplicate/legacy baseline IDs are forbidden in new Formal evidence"
        )

    hashes = {name: sha256_file(path) for name, path in sorted(artefacts.items())}
    canonical = json.dumps(
        {"rtc_source_tree_sha256": current_code_sha, "artefacts": hashes},
        sort_keys=True,
        separators=(",", ":"),
    )
    policy_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "contract": "WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND",
        "policy_sha256": policy_sha,
        "rtc_source_tree_sha256": current_code_sha,
        "physical_network_sha256": physical_contract_sha256(
            artefacts["frozen_inp"]
        ),
        "objective_contract": "TFV_PRIMARY__PRIORITY_PFV_SOFT_SECONDARY_V1",
        "priority_is_hard_constraint": False,
        "priority_nodes": list(priority),
        "sensor_nodes": list(sensors),
        "causal_timing": timing_payload,
        "model_contracts": model_contracts,
        "rainfall_design": rainfall_design,
        "no_control_contract": no_control,
        "runtime_acceptance": runtime_acceptance,
        "formal_strategy_matrix": list(strategies),
        "formal_metric_aggregation": "equal_weight_per_independent_rainfall_group",
        "fresh_workspace": fresh_workspace,
        "fresh_data_requirement": (
            "all hydraulic trajectories, D1/D2/D3 branches, trained models and closed-loop "
            "evidence are generated under the bound Fresh Workspace and current source tree; "
            "historical/stale-code RTC outputs/models are not admissible"
        ),
        "artefacts": artefacts,
        "sha256": hashes,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create code/time/data-bound causal TFV-first Policy Lock"
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = create_formal_policy_lock_v5(
        ledger_path=args.ledger,
        artefacts_path=args.artifacts,
        output_path=args.out,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
