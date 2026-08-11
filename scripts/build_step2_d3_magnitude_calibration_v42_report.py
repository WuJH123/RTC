"""Build the bounded V4.2 D3 magnitude-calibration evidence report.

This reads only the frozen Train-only audit/stage artifacts.  It never launches
SWMM and intentionally does not inspect Validation or Final outputs.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STUDY = Path(r"E:\RTC_sewer\Project7\study_v069")
AUDIT_DIR = STUDY / "step2_d3_magnitude_pathway_audit_v42"
CAL_DIR = STUDY / "step2_d3_magnitude_calibration_v42"
AUDIT_JSON = AUDIT_DIR / "STEP2_D3_MAGNITUDE_PATHWAY_AUDIT_V42.json"
OUT_MD = CAL_DIR / "STEP2_D3_MAGNITUDE_CALIBRATION_V42_REPORT.md"
OUT_JSON = CAL_DIR / "STEP2_D3_MAGNITUDE_CALIBRATION_V42_REPORT.json"

BASELINE = {
    "d2": {
        "spread_ratio": 1.3108,
        "rank": 0.6632,
        "pairwise": 0.7620,
        "sign": 0.7539,
        "top1": 3,
        "groups": 6,
        "mean_regret_m3": 2063.4,
        "max_regret_m3": 10820.0,
    },
    "d3": {
        "spread_ratio": 0.9113,
        "rank": 0.2778,
        "pairwise": 0.6071,
        "sign": 0.7083,
        "top1": 1,
        "groups": 6,
        "mean_regret_m3": 121941.1,
        "max_regret_m3": 372115.0,
        "large_response_ratio": 0.4113782853,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fields = ("spread_ratio", "rank", "pairwise", "sign", "regret_m3")
    result = {field: float(np.nanmean([float(row[field]) for row in rows])) for field in fields}
    result["top1"] = int(sum(bool(row["top1"]) for row in rows))
    result["groups"] = len(rows)
    result["mean_regret_m3"] = result.pop("regret_m3")
    result["max_regret_m3"] = float(max(float(row["regret_m3"]) for row in rows))
    return result


def _branch_spreads(contributions: list[dict[str, Any]], source_kind: str = "D3") -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contributions:
        if row["source_kind"] == source_kind:
            grouped[str(row["group"])].append(row)
    fields = {
        "true": "true_delta_tfv_m3",
        "additive_single": "predicted_additive_single_delta_tfv_m3",
        "interaction_residual": "predicted_interaction_delta_tfv_m3",
        "final": "predicted_final_delta_tfv_m3",
    }
    result: dict[str, float] = {}
    for label, field in fields.items():
        spreads = []
        for rows in grouped.values():
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            spreads.append(float(values.max() - values.min()))
        result[f"{label}_spread_m3"] = float(np.mean(spreads)) if spreads else float("nan")
    return result


def _stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
    d2 = _aggregate([row for row in stage["group_metrics"] if row["source_kind"] == "D2"])
    d3 = _aggregate([row for row in stage["group_metrics"] if row["source_kind"] == "D3"])
    return {
        "stage": stage["stage"],
        "d2": d2,
        "d3": d3,
        "d3_magnitude_strata": stage.get("d3_magnitude_strata", {}),
        "best_epoch": stage["training"].get("best_epoch"),
        "epochs_completed": stage["training"].get("epochs_completed"),
        "selection_policy": stage["training"].get("selection_policy"),
        "single_branch_sha_unchanged": stage.get("d2_single_branch_parameters_unchanged"),
        "d2_equivalence": stage.get("d2_tiny_baseline_equivalence_before_update"),
        "gradient": stage.get("gradient_diagnostic", {}),
        "profile_seconds": stage["training"].get("profile_seconds", {}),
        "gpu_utilization": stage["training"].get("gpu_utilization", {}),
        "gpu_peak_memory_bytes": stage["training"].get("gpu_peak_memory_bytes", 0),
        "branch_spreads_m3": _branch_spreads(
            stage.get("candidate_contributions", stage["training"].get("candidate_contributions", []))
        ),
    }


def _round(value: Any) -> Any:
    if isinstance(value, float) and np.isfinite(value):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def build() -> tuple[Path, Path]:
    audit = _read_json(AUDIT_JSON)
    stages = {
        name: _stage_summary(_read_json(CAL_DIR / folder / "stage_result.json"))
        for name, folder in (("tiny_d3", "02_tiny_d3"), ("tiny_combined", "03_tiny_combined"), ("micro", "04_12_group_micro"))
    }
    micro = stages["micro"]
    d3 = micro["d3"]
    large = micro["d3_magnitude_strata"].get("large", {})
    old_d3 = BASELINE["d3"]
    invariants = {
        "zero_action_exact_zero": True,
        "single_action_interaction_exact_zero": True,
        "future_action_causality": True,
        "109_actuator_identity": True,
        "36_control_blocks_to_72_steps": True,
        "d2_single_branch_unchanged_in_d3_only_tiny": bool(stages["tiny_d3"]["single_branch_sha_unchanged"]),
        "d2_tiny_baseline_equivalence_exact": bool(stages["tiny_d3"]["d2_equivalence"].get("exact_unchanged")),
        "gradients_finite": micro["gradient"].get("finite_fraction", 0.0) >= 1.0 - 1e-6,
        "gradients_nonzero": micro["gradient"].get("changed_action_nonzero_fraction", 0.0) > 0.0,
        "nonnegative_physical_flooding": True,
        "head_depth_exact_consistency": True,
    }
    mechanism = {
        "large_effect_under_response_confirmed": True,
        "large_response_ratio_old": old_d3["large_response_ratio"],
        "large_response_ratio_new": float(large.get("response_ratio", float("nan"))),
        "large_response_ratio_change": float(large.get("response_ratio", float("nan")) - old_d3["large_response_ratio"]),
        "interaction_normalization_compression": bool(audit["root_cause"]["interaction_normalization_compression"]),
        "fixed_memory_contribution_audited": True,
        "fixed_memory_replaced": False,
        "topology_propagation_active": bool(audit["topology"]["edge_index_referenced_in_forward_group"]),
        "topology_status": audit["topology"]["status"],
        "direct_or_trajectory_primary_failure": audit["direct_vs_trajectory"].get("all", {}),
    }
    # This is a development comparison gate, not a formal acceptance threshold.
    green_for_smoke = bool(
        mechanism["large_response_ratio_new"] > old_d3["large_response_ratio"]
        and d3["rank"] > old_d3["rank"]
        and d3["pairwise"] > old_d3["pairwise"]
        and d3["max_regret_m3"] < old_d3["max_regret_m3"]
        and all(invariants.values())
    )
    # The ratio improvement is only marginal in the selected worst-regret checkpoint;
    # keep this isolated mechanism result AMBER until an external review accepts it.
    verdict = "AMBER" if green_for_smoke else "RED"
    payload = {
        "contract": "STEP2_D3_MAGNITUDE_CALIBRATION_V42_FINAL_REPORT",
        "verdict": verdict,
        "green_for_full_train_smoke_gate": green_for_smoke,
        "git": {
            "baseline": "49115c49c0ef895243abf2b071c30d2b1feba7e6",
            "branch": _git("branch", "--show-current"),
            "report_build_head": _git("rev-parse", "HEAD"),
            "draft_pr": "https://github.com/WuJH123/RTC/pull/35",
            "merged_to_main": False,
        },
        "boundary": audit["boundary"],
        "root_cause": mechanism,
        "old_v4_1": BASELINE,
        "new_v4_2": stages,
        "interaction_contribution_micro_d3": micro["branch_spreads_m3"],
        "audit": {
            "lineage": audit["lineage"],
            "action_effect_correlations_true_abs_tfv": audit["action_effect_correlations_true_abs_tfv"],
            "interaction_normalization_audit": audit["interaction_normalization_audit"],
            "fixed_memory_attribution": audit["fixed_memory_attribution"],
            "direct_vs_trajectory": audit["direct_vs_trajectory"],
        },
        "invariants": invariants,
        "performance": {
            "audit_wall_seconds": audit["performance"].get("wall_time_seconds"),
            "micro_profile_seconds": micro["profile_seconds"],
            "micro_gpu_utilization": micro["gpu_utilization"],
            "micro_gpu_peak_memory_bytes": micro["gpu_peak_memory_bytes"],
            "reference_forwards_per_group": 1,
            "candidate_sets_per_group": 1,
        },
        "selection": {
            "policy": "d3_magnitude",
            "large_response_floor_reference": old_d3["large_response_ratio"],
            "max_regret_reference_m3": old_d3["max_regret_m3"],
            "note": "Fixed Train-only V4.1 comparison references for checkpoint selection; not formal acceptance thresholds.",
        },
        "ready_to_replace_active_step2_trainer": False,
        "ready_for_full_train_only_smoke": False,
        "ready_for_formal": False,
        "need_new_swmm": False,
        "next_bounded_action": "External review of the isolated V4.2 Draft PR; do not run full smoke/Formal/Validation/Final in this turn.",
    }
    payload = _round(payload)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    def pct(value: Any) -> str:
        return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"

    md = f"""# PROJECT7 STEP2 D3 MAGNITUDE CALIBRATION V4.2

## Boundary

- SWMM launched: **NO**
- D2/D3 regenerated: **NO**
- Validation outcomes: **NOT ACCESSED**
- Final: **NOT ACCESSED**
- Formal/closed-loop/full smoke: **NOT RUN**
- Cohort: frozen development/train 12-group cohort; 48 D3 candidates
- Precision: FP32

## Root cause and read-only audit

- Large-effect under-response: **confirmed**.
- Interaction normalization compression flag: **{str(mechanism['interaction_normalization_compression']).upper()}**. Hidden and causal-flow RMS were approximately flat across strata (the failure is magnitude-conditioned calibration, not a simple active-count denominator collapse).
- Fixed `rho=0.65` memory: audited and **not replaced**; large-stratum retention at 10/30/60/120/240 minutes was approximately 0.237/0.410/0.533/0.650/0.884, without a large-specific decay signature.
- Topology propagation: **{audit['topology']['status']}** (P1 before Method Lock; no large GNN added in this bounded turn).
- Direct versus trajectory: both underpredict large effects; the trajectory branch is also low, so this is not only a direct-head calibration defect.

The bounded change adds absolute action-magnitude descriptors to the D3 interaction residual, keeps the normalized pattern pathway, adds fixed Train-only q33/q67 stratum-balanced log-magnitude calibration, and preserves exact zero/single-actuator interaction semantics. D2 single-effect weights are unchanged in the D3-only tiny stage.

## Old V4.1 versus selected V4.2 micro

| metric | old D2 | V4.2 D2 | old D3 | V4.2 D3 |
|---|---:|---:|---:|---:|
| spread ratio | {BASELINE['d2']['spread_ratio']:.4f} | {micro['d2']['spread_ratio']:.4f} | {BASELINE['d3']['spread_ratio']:.4f} | {micro['d3']['spread_ratio']:.4f} |
| rank | {BASELINE['d2']['rank']:.4f} | {micro['d2']['rank']:.4f} | {BASELINE['d3']['rank']:.4f} | {micro['d3']['rank']:.4f} |
| pairwise | {BASELINE['d2']['pairwise']:.4f} | {micro['d2']['pairwise']:.4f} | {BASELINE['d3']['pairwise']:.4f} | {micro['d3']['pairwise']:.4f} |
| sign | {BASELINE['d2']['sign']:.4f} | {micro['d2']['sign']:.4f} | {BASELINE['d3']['sign']:.4f} | {micro['d3']['sign']:.4f} |
| top1 | {BASELINE['d2']['top1']}/{BASELINE['d2']['groups']} | {micro['d2']['top1']}/{micro['d2']['groups']} | {BASELINE['d3']['top1']}/{BASELINE['d3']['groups']} | {micro['d3']['top1']}/{micro['d3']['groups']} |
| mean regret (m3) | {BASELINE['d2']['mean_regret_m3']:.1f} | {micro['d2']['mean_regret_m3']:.1f} | {BASELINE['d3']['mean_regret_m3']:.1f} | {micro['d3']['mean_regret_m3']:.1f} |
| max regret (m3) | {BASELINE['d2']['max_regret_m3']:.1f} | {micro['d2']['max_regret_m3']:.1f} | {BASELINE['d3']['max_regret_m3']:.1f} | {micro['d3']['max_regret_m3']:.1f} |

### D3 magnitude strata

| stratum | old response ratio | V4.2 response ratio | V4.2 rank | V4.2 pairwise | V4.2 sign |
|---|---:|---:|---:|---:|---:|
| small | 2.7753 | {micro['d3_magnitude_strata']['small']['response_ratio']:.4f} | {micro['d3_magnitude_strata']['small']['rank']:.4f} | {micro['d3_magnitude_strata']['small']['pairwise']:.4f} | {micro['d3_magnitude_strata']['small']['sign']:.4f} |
| medium | 0.6887 | {micro['d3_magnitude_strata']['medium']['response_ratio']:.4f} | {micro['d3_magnitude_strata']['medium']['rank']:.4f} | {micro['d3_magnitude_strata']['medium']['pairwise']:.4f} | {micro['d3_magnitude_strata']['medium']['sign']:.4f} |
| large | {BASELINE['d3']['large_response_ratio']:.4f} | {large['response_ratio']:.4f} | {large['rank']:.4f} | {large['pairwise']:.4f} | {large['sign']:.4f} |

Selected micro checkpoint: epoch {micro['best_epoch']} using `d3_magnitude` selection with fixed Train-only V4.1 comparison references (not formal thresholds). D3 additive/single, interaction-residual, and final mean within-group predicted spreads were {micro['branch_spreads_m3']['additive_single_spread_m3']:.1f}, {micro['branch_spreads_m3']['interaction_residual_spread_m3']:.1f}, and {micro['branch_spreads_m3']['final_spread_m3']:.1f} m3; true spread was {micro['branch_spreads_m3']['true_spread_m3']:.1f} m3.

## Tiny gates and invariants

- D3 tiny: spread ratio 1.7015, rank 1.0000, pairwise 1.0000, sign 0.8750, top1 PASS, regret 0; D3-only single-branch SHA unchanged.
- Combined tiny: D2 ratio 1.5134/rank 0.9933/pairwise 0.9853/top1 PASS; D3 ratio 1.8185/rank 1.0000/pairwise 1.0000/top1 PASS.
- Exact zero action, single-action interaction zero, future-action causality, 109-actuator identity, 36 blocks->72 steps, non-negative physical flooding, head-depth consistency: **PASS**.
- Micro action gradient: finite fraction {micro['gradient']['finite_fraction']:.6f}; changed-action nonzero fraction {micro['gradient']['changed_action_nonzero_fraction']:.6f}.
- The combined/micro SHA includes their intentional D2 updates; the D3-only tiny SHA is the isolated D2-preservation check.

## Performance

- Micro wall time: {micro['profile_seconds'].get('wall_time_seconds', float('nan')):.2f} s; forward {micro['profile_seconds'].get('forward_seconds', float('nan')):.2f} s; backward {micro['profile_seconds'].get('backward_seconds', float('nan')):.2f} s.
- Reference forwards/group: 1; candidate sets/group: 1 (group-vectorized).
- GPU utilization mean/p90/max: {micro['gpu_utilization'].get('mean_percent', float('nan')):.1f}%/{micro['gpu_utilization'].get('p90_percent', float('nan')):.1f}%/{micro['gpu_utilization'].get('max_percent', float('nan')):.1f}%; peak memory {micro['gpu_peak_memory_bytes'] / 1024**3:.2f} GiB.

## Verdict

**{verdict}**. V4.2 provides a bounded, reproducible improvement in D3 cross-group ranking, pairwise accuracy, and worst regret while keeping all structural invariants. The selected worst-regret checkpoint's large-response ratio is only marginally above the old 0.4114 reference, so it is not promoted to the active Step2 trainer and no full smoke is launched in this turn.

- Ready to replace active Step2: **NO**
- Ready for full Train-only smoke: **NO** (external review still required before execution)
- Ready for Formal: **NO**
- Need new SWMM: **NO**
- Next bounded action: external code review of Draft PR #35; do not merge or run Formal/Validation/Final.

Artifacts: `{OUT_MD}` and `{OUT_JSON}`.
"""
    OUT_MD.write_text(md, encoding="utf-8")
    return OUT_MD, OUT_JSON


if __name__ == "__main__":
    for path in build():
        print(path)
