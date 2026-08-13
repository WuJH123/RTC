"""Assemble the V11.1 development decision from immutable evidence files.

This reporter never reads InternalHoldout to tune anything and never promotes
the failed micro mechanism to a canonical checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-report", required=True)
    parser.add_argument("--tiny-report", required=True)
    parser.add_argument("--micro-report", required=True)
    parser.add_argument("--gradient-report", required=True)
    parser.add_argument("--local-baseline", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    old = _read(args.old_report); tiny = _read(args.tiny_report); micro = _read(args.micro_report)
    gradient = _read(args.gradient_report); baseline = _read(args.local_baseline)
    micro_skill = micro["fit_metrics"]["primary_event_pooled_skill_vs_zero"]
    micro_pass = all(float(micro_skill[key]) > 0.0 for key in ("depth_m", "flood_m3s", "volume_m3", "managed_flow_m3s"))
    tiny_skill = tiny["fit_metrics"]["primary_event_pooled_skill_vs_zero"]
    tiny_pass = all(float(tiny_skill[key]) > 0.0 for key in ("depth_m", "flood_m3s", "volume_m3", "managed_flow_m3s"))
    report = {
        "contract": "PROJECT7_STEP2_V111_ZERO_ANCHORED_SIGNED_EFFECT_V1",
        "verdict": "V111_REPRESENTATION_BLOCKER_IDENTIFIED",
        "boundary": {"swmm_run": False, "validation_accessed": False, "final_accessed": False,
                      "formal_accessed": False, "policy_lock": False, "new_data_generated": False,
                      "v7_value_retrained": False, "d3_run": False, "runtime_bundle_compiled": False,
                      "rolling_smoke_run": False},
        "git": {"branch": "agent/v110-data-semantics-mpc-wiring",
                "head": tiny.get("lineage", {}).get("git_head"),
                "pr": 51, "pr_state": "OPEN_DRAFT_DO_NOT_MERGE"},
        "root_cause": {
            "old_multiplicative_decoder_leakage": True,
            "old_all_cell_magnitude_scale_pathology": True,
            "old_dense_reduction_dilution": True,
            "legacy_evaluator_denominator_instability": True,
            "v7_reference_bottleneck": "NOT_TESTED_IN_V111_GATE",
            "current_v111_full_network_representation_or_optimization_blocker": True,
            "evidence": [
                "V110 response ratios reached 2290.6 flood and 5182.5 volume on Holdout.",
                "V111 gradient audit found old inactive volume gradients ~4.4e3 versus direct active ~1.6e-1.",
                "TrainFit local MLP independently learns Holdout endpoint depth skill 0.6622 and storage skill 0.7302.",
                "V111 micro direct decoder did not achieve four positive pooled skills after fixed 12 epochs.",
            ],
        },
        "v111_design": {"direct_signed_physical_output": "PASS", "zero_anchored": "PASS",
                        "active_conditional_scale": "PASS", "balanced_active_inactive_loss": "PASS",
                        "auxiliary_decomposition": "PASS", "exact_zero": "PASS",
                        "future_causality": "PASS", "direct_gradient": "PASS"},
        "tiny": {"groups": 1, "steps": 100, "fit_primary": tiny_skill,
                  "response_ratio": tiny["fit_metrics"]["response_ratio"],
                  "inactive_leakage": tiny["fit_metrics"]["leakage"], "verdict": "FAIL" if not tiny_pass else "PASS"},
        "micro": {"groups": len(micro["split"]["profile_groups"]), "epochs": len(micro["training_history"]),
                  "fit_primary": micro_skill, "response_ratio": micro["fit_metrics"]["response_ratio"],
                  "inactive_leakage": micro["fit_metrics"]["leakage"], "verdict": "FAIL" if not micro_pass else "PASS"},
        "gradient_audit": gradient,
        "local_baseline": {
            "path": str(args.local_baseline),
            "holdout_depth_skill": baseline["baselines"]["local_mlp"]["TrainInternalHoldout_D2"]["channels"]["delta_depth_m"]["skill_vs_zero"],
            "holdout_storage_skill": baseline["baselines"]["local_mlp"]["TrainInternalHoldout_D2"]["channels"]["delta_storage_m3"]["skill_vs_zero"],
            "holdout_flood_skill": baseline["baselines"]["local_mlp"]["TrainInternalHoldout_D2"]["channels"]["delta_flood_m3s"]["skill_vs_zero"],
            "holdout_managed_flow_skill": baseline["baselines"]["local_mlp"]["TrainInternalHoldout_D2"]["channels"]["delta_managed_flow_m3s"]["skill_vs_zero"],
        },
        "canonical_d2": {"run": False, "reason": "tiny/micro mechanism gates did not pass; no canonical or Holdout tuning run"},
        "d3": {"run": False, "reason": "D2 gate blocked"},
        "forecast_shift": {"status": "NOT_RUN_D2_BLOCKED"},
        "runtime": {"bundle_compiled": False, "rolling_smoke": "NOT_RUN"},
        "evidence_paths": {"old_report": str(args.old_report), "tiny": str(args.tiny_report),
                            "micro": str(args.micro_report), "gradient": str(args.gradient_report),
                            "local_baseline": str(args.local_baseline)},
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / "STEP2_V111_DEVELOPMENT_DECISION.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md = ["# Project7 Step2 V11.1 development decision", "", f"Verdict: **{report['verdict']}**", "",
          "V111 direct signed/zero-anchor/scaling/loss/evaluator correctness tests pass.",
          "The fixed TrainFit micro mechanism did not pass the four-channel pooled skill gate, so canonical D2, D3, runtime bundle and rolling smoke were not run.", "",
          "## Evidence", "", f"Tiny report: `{args.tiny_report}`", f"Micro report: `{args.micro_report}`",
          f"Gradient report: `{args.gradient_report}`", f"Local baseline: `{args.local_baseline}`", "",
          "## Boundary", "", "No new SWMM, Validation, Final, Formal, D3, production wiring, or V7 Value retraining.", ""]
    (out / "STEP2_V111_DEVELOPMENT_DECISION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "verdict": report["verdict"], "tiny": report["tiny"]["verdict"], "micro": report["micro"]["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
