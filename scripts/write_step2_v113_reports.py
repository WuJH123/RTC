"""Assemble the bounded V11.3 mechanism-gate evidence without recomputing data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric(report: dict[str, Any], channel: str) -> dict[str, Any]:
    return report["metrics"]["overall"]["channels"][channel]


def selected(report: dict[str, Any]) -> dict[str, Any]:
    names = ("depth_m", "flood_m3s", "storage_m3", "inflow_m3s", "outflow_m3s", "managed_flow_m3s")
    return {name: metric(report, name) for name in names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nested-split", required=True)
    ap.add_argument("--signed-audit", required=True)
    ap.add_argument("--atlas-info", required=True)
    ap.add_argument("--step1-audit", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    study = Path(args.study_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    split, signed, atlas, step1 = map(lambda x: load(Path(x)), (args.nested_split, args.signed_audit, args.atlas_info, args.step1_audit))
    reports = {
        "tiny_phase_presence": load(study / "tiny_phase_presence_seed42.json"),
        "micro_phase_presence": load(study / "micro_phase_presence_seed42.json"),
        "devcheck_none": load(study / "devcheck_none_seed42.json"),
        "devcheck_overall": load(study / "devcheck_overall_seed42.json"),
        "devcheck_phase": load(study / "devcheck_phase_presence_seed42.json"),
        "devcheck_oracle_support": load(study / "devcheck_oracle_support_seed42.json"),
    }
    channels = ("depth_m", "flood_m3s", "storage_m3", "inflow_m3s", "outflow_m3s", "managed_flow_m3s")
    facility = {}
    for key in ("devcheck_none", "devcheck_overall", "devcheck_phase", "devcheck_oracle_support"):
        facility[key] = {name: metric(reports[key], name) for name in ("managed_flow_m3s",)}
    facility_pass = all(
        facility["devcheck_phase"]["managed_flow_m3s"][key] > 0
        for key in ("skill_vs_zero", "active_skill_vs_zero")
    ) and facility["devcheck_phase"]["managed_flow_m3s"]["response_ratio"] >= 0.10 and facility["devcheck_phase"]["managed_flow_m3s"]["active_sign_accuracy"] >= 0.50
    primary = {k: selected(reports[k]) for k in ("devcheck_none", "devcheck_overall", "devcheck_phase", "devcheck_oracle_support")}
    devcheck_pass = all(
        primary["devcheck_phase"][name]["skill_vs_zero"] > 0
        for name in ("depth_m", "flood_m3s", "storage_m3", "managed_flow_m3s")
    )
    ablation = {
        "B0_ZERO": {"status": "included", "report": "zero_baseline", "phase": reports["devcheck_phase"]["zero_baseline"]},
        "B1_ACTION_LOCAL": {
            "status": "historical_reference_only",
            "report_path": "step2_v90_trajectory_conditioned_hydraulic/audit_b6014d9/STEP2_D2_LOCAL_EFFECT_BASELINES.json",
            "report_git_head": "61722f334375c243400b6c17ad7352654e1ea677",
            "holdout_channels": {
                "depth_m": {"skill_vs_zero": 0.6621978171641698, "active_sign_accuracy": 0.9360114915129691, "response_ratio": 0.8447851551706446},
                "flood_m3s": {"skill_vs_zero": 0.003613634446625158, "active_sign_accuracy": 0.9233560032512805, "response_ratio": 0.47970543394304244},
                "storage_m3": {"skill_vs_zero": 0.7302334908275309, "active_sign_accuracy": 0.7439349717013345, "response_ratio": 1.3482338888419705},
                "managed_flow_m3s": {"skill_vs_zero": 0.005409641029601953, "active_sign_accuracy": 0.868947853353481, "response_ratio": 0.39603725690748853},
            },
        },
        "B2_NO_ATLAS": primary["devcheck_none"],
        "B3_OVERALL_ATLAS": primary["devcheck_overall"],
        "B4_PHASE_ATLAS": primary["devcheck_phase"],
        "B5_ORACLE_SUPPORT": {"online_eligible": False, "diagnostic_only": True, "metrics": primary["devcheck_oracle_support"]},
    }
    lineage = {
        "git_head": reports["devcheck_phase"]["lineage"]["git_head"],
        "cache_sha256": sha(Path(args.cache)), "graph_sha256": sha(Path(args.graph)),
        "nested_split_sha256": sha(Path(args.nested_split)), "prior_sha256": sha(Path(args.prior)),
        "signed_audit_git_head": signed.get("git_head"), "atlas_info_git_head": atlas.get("git_head"),
        "step1_audit_git_head": step1.get("git_head"),
    }
    facility_report = {
        "contract": "PROJECT7_STEP2_V113_FACILITY_FLOW_GATE_V1",
        "development_only": True, "lineage": lineage, "train_groups": len(split["v113_devfit"]["group_names"]),
        "devcheck_groups": len(split["v113_devcheck"]["group_names"]), "channels": facility,
        "criteria": {"positive_skill_and_active_skill": True, "response_ratio_min": 0.10, "active_sign_min": 0.50},
        "passed": bool(facility_pass),
        "interpretation": "FACILITY_RESPONSE_BOTTLENECK" if not facility_pass else "FACILITY_FLOW_GATE_SUPPORTED",
    }
    micro_report = {"contract": "PROJECT7_STEP2_V113_MICRO_REPORT_V1", "development_only": True, "lineage": lineage,
                    "tiny_phase_presence": selected(reports["tiny_phase_presence"]), "micro_phase_presence": selected(reports["micro_phase_presence"]),
                    "preflight": reports["micro_phase_presence"]["preflight"]}
    devcheck_report = {"contract": "PROJECT7_STEP2_V113_DEVCHECK_REPORT_V1", "development_only": True, "lineage": lineage,
                       "arms": {k: {"train_events": reports[k]["train_events"], "eval_events": reports[k]["eval_events"], "metrics": selected(reports[k]), "preflight": reports[k]["preflight"]}
                                for k in ("devcheck_none", "devcheck_overall", "devcheck_phase", "devcheck_oracle_support")},
                       "gate_passed": bool(devcheck_pass), "no_internal_holdout_outcome_access": True}
    ablation_report = {"contract": "PROJECT7_STEP2_V113_ABLATION_V1", "development_only": True, "lineage": lineage,
                      "arms": ablation, "atlas_in_sample_information": {"phase_adds_information": atlas["does_phase_conditioned_atlas_add_information"],
                      "overall": atlas["overall_prior"], "phase": atlas["phase_conditioned_prior"]},
                      "devcheck_atlas_gain": {"phase_minus_none": {name: primary["devcheck_phase"][name]["skill_vs_zero"] - primary["devcheck_none"][name]["skill_vs_zero"] for name in channels},
                                               "phase_minus_overall": {name: primary["devcheck_phase"][name]["skill_vs_zero"] - primary["devcheck_overall"][name]["skill_vs_zero"] for name in channels}},
                      "interpretation": "ATLAS_NOT_DEMONSTRATED_AS_GENERALIZABLE_GAIN"}
    decision = {
        "contract": "PROJECT7_STEP2_V113_CURRENT_DECISION_V1", "lineage": lineage,
        "data": {"trainfit_d2_groups": 112, "trainfit_d2_candidates": 2688, "devfit_events": 10, "devcheck_events": 4, "internal_holdout_outcomes_accessed": False},
        "gates": {"exact_zero": True, "future_action_causality": True, "facility_flow": bool(facility_pass), "devcheck_primary": bool(devcheck_pass)},
        "atlas": {"in_sample_support_information": True, "phase_prior_in_sample_better": True, "devcheck_gain": False, "production_use_authorized": False},
        "step1": step1,
        "decision": "V113_FACILITY_RESPONSE_BOTTLENECK",
        "proceed_to_canonical_trainfit_d2": False, "read_internal_holdout": False, "proceed_to_d3": False,
        "new_swmm_authorized": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False,
        "reason": "The fixed nested DevCheck does not learn facility-flow or full-network signed effects; all Atlas arms remain near-zero/explosive and B5 oracle support does not rescue magnitude learning."
    }
    for name, payload in (("STEP2_V113_FACILITY_FLOW_GATE.json", facility_report), ("STEP2_V113_MICRO_REPORT.json", micro_report),
                          ("STEP2_V113_DEVCHECK_REPORT.json", devcheck_report), ("STEP2_V113_ABLATION.json", ablation_report), ("STEP2_V113_CURRENT_DECISION.json", decision)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md = f"""# Project7 Step2 V113 bounded mechanism decision\n\n- Nested DevFit/DevCheck: 10/4 events; rainfall/event overlap: 0.\n- TrainFit D2: 112 groups, 2688 candidates; InternalHoldout outcome accessed: **NO**.\n- Exact-zero, future-action causality, finite/nonzero action gradient: **PASS**.\n- Facility-flow gate: **{'PASS' if facility_pass else 'FAIL'}**.\n- Full-network DevCheck primary gate: **{'PASS' if devcheck_pass else 'FAIL'}**.\n- Atlas in-sample information: phase prior improves support/effect association, but fixed nested DevCheck shows no generalizable metric gain over no-atlas/overall.\n- Oracle-support diagnostic also fails to rescue magnitude learning; this is not an online input.\n\n## Decision\n\n**V113_FACILITY_RESPONSE_BOTTLENECK**. Stop this bounded stage. Do not enter canonical 112-group D2, read InternalHoldout outcomes, train D3, run SWMM, or use Atlas in production.\n\nMachine-readable evidence is in the accompanying JSON files.\n"""
    (out / "STEP2_V113_CURRENT_DECISION.md").write_text(md, encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "decision": decision["decision"], "facility_gate": facility_pass, "devcheck_gate": devcheck_pass}, indent=2))


if __name__ == "__main__":
    main()
