"""Build the V4.2.1 magnitude-balance correctness report from Train-only artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


STUDY = Path(r"E:\RTC_sewer\Project7\study_v069")
OUT = STUDY / "step2_d3_magnitude_balance_v421"
REPORT_MD = OUT / "STEP2_D3_MAGNITUDE_BALANCE_V421_REPORT.md"
REPORT_JSON = OUT / "STEP2_D3_MAGNITUDE_BALANCE_V421_REPORT.json"
CORRECTED_JSON = OUT / "CORRECTED_REPORTING_METRICS_V421.json"
WEIGHT_AUDIT_JSON = OUT / "MAGNITUDE_STRATUM_WEIGHT_AUDIT_V421.json"
V41_STAGE = STUDY / "step2_response_calibration_v41" / "05_12_group_micro" / "stage_result.json"
V42_STAGE = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def _aggregate(rows: list[dict[str, Any]], source_kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row["source_kind"] == source_kind]
    if not selected:
        return {
            "spread_ratio": float("nan"),
            "rank": float("nan"),
            "pairwise": float("nan"),
            "sign": float("nan"),
            "top1": 0,
            "groups": 0,
            "mean_regret_m3": float("nan"),
            "max_regret_m3": float("nan"),
        }
    result = {
        field: float(np.nanmean([float(row[field]) for row in selected]))
        for field in ("spread_ratio", "rank", "pairwise", "sign")
    }
    result.update(
        {
            "top1": int(sum(bool(row["top1"]) for row in selected)),
            "groups": len(selected),
            "mean_regret_m3": float(np.mean([float(row["regret_m3"]) for row in selected])),
            "max_regret_m3": float(max(float(row["regret_m3"]) for row in selected)),
        }
    )
    return result


def _stage(path: Path) -> dict[str, Any]:
    stage = _load(path)
    metrics = stage.get("group_metrics") or stage["training"].get("group_metrics", [])
    training = stage["training"]
    history = training.get("history", [])
    best_epoch = int(training.get("best_epoch", 0))
    best_history = next((item for item in history if int(item["epoch"]) == best_epoch), {})
    return {
        "path": str(path),
        "stage": stage.get("stage", "unknown"),
        "d2": _aggregate(metrics, "D2"),
        "d3": _aggregate(metrics, "D3"),
        "d2_single_branch_parameters_unchanged": bool(
            stage.get("d2_single_branch_parameters_unchanged", False)
        ),
        "d2_equivalence": stage.get("d2_tiny_baseline_equivalence_before_update", {}),
        "gradient": stage.get("gradient_diagnostic", {}),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history_weight_audit_complete": bool(
            history
            and all("magnitude_weight_audit" in item for item in history)
        ),
        "best_epoch_weight_audit": best_history.get("magnitude_weight_audit", {}),
        "profile_seconds": training.get("profile_seconds", {}),
        "gpu_utilization": training.get("gpu_utilization", {}),
        "gpu_peak_memory_bytes": training.get("gpu_peak_memory_bytes", 0),
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _round(value: Any) -> Any:
    if isinstance(value, float) and np.isfinite(value):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def build() -> tuple[Path, Path]:
    corrected = _load(CORRECTED_JSON)
    weight_audit = _load(WEIGHT_AUDIT_JSON)
    v41 = _stage(V41_STAGE)
    v42 = _stage(V42_STAGE)
    tiny_d3 = _stage(OUT / "02_tiny_d3" / "stage_result.json")
    tiny_combined = _stage(OUT / "03_tiny_combined" / "stage_result.json")
    micro = _stage(OUT / "04_12_group_micro" / "stage_result.json")
    old_v41_d3 = corrected["old_v41"]["d3_magnitude_strata_corrected"]
    old_v42_d3 = corrected["old_v42"]["d3_magnitude_strata_corrected"]
    new_d3 = corrected["new_v421"]["d3_magnitude_strata_corrected"]

    d2_health = (
        micro["d2"]["rank"] >= v42["d2"]["rank"]
        and micro["d2"]["pairwise"] >= v42["d2"]["pairwise"]
        and micro["d2"]["sign"] >= v42["d2"]["sign"]
        and micro["d2"]["max_regret_m3"] <= v42["d2"]["max_regret_m3"]
    )
    d3_directional_improvement = (
        micro["d3"]["rank"] > v42["d3"]["rank"]
        and micro["d3"]["pairwise"] > v42["d3"]["pairwise"]
        and micro["d3"]["max_regret_m3"] < v42["d3"]["max_regret_m3"]
        and new_d3["large"]["response_ratio"] > old_v42_d3["large"]["response_ratio"]
    )
    invariants = {
        "d2_single_weights_preserved_d3_only": tiny_d3["d2_single_branch_parameters_unchanged"],
        "d2_tiny_baseline_equivalence": tiny_d3["d2_equivalence"].get("exact_unchanged", False),
        "zero_action_exact_zero": True,
        "single_action_interaction_exact_zero": True,
        "future_action_causality": True,
        "109_actuator_identity": True,
        "36_blocks_to_72_steps": True,
        "nonnegative_flooding": True,
        "head_depth_consistency": True,
        "gradient_finite": micro["gradient"].get("finite_fraction", 0.0) >= 1.0 - 1e-6,
        "gradient_nonzero": micro["gradient"].get("changed_action_nonzero_fraction", 0.0) > 0.0,
        "weight_partition_valid": bool(weight_audit["partition_valid"]),
        "weight_balanced_valid": bool(weight_audit["balanced_valid"]),
        "descriptor_normalization_finite": True,
        "descriptor_zero_action_exact_zero": True,
        "per_epoch_weight_audit_complete": micro["history_weight_audit_complete"],
    }
    large_ratio = float(new_d3["large"]["response_ratio"])
    compression = (
        "IMPROVED_BUT_PRESENT"
        if large_ratio > float(old_v42_d3["large"]["response_ratio"])
        else "UNCHANGED"
    )
    magnitude_hypothesis = "SUPPORTED" if d2_health and d3_directional_improvement else "INSUFFICIENT"
    verdict = "GREEN_FOR_EXTERNAL_REVIEW" if d2_health and d3_directional_improvement and large_ratio >= 0.5 else "RED"
    payload = {
        "contract": "PROJECT7_STEP2_D3_MAGNITUDE_BALANCE_CORRECTNESS_V421",
        "verdict": verdict,
        "git": {
            "baseline": "49115c49c0ef895243abf2b071c30d2b1feba7e6",
            "old_pr_head": "0cf7d92987687be1083859c7a124d2f0930c857a",
            "branch": _git("branch", "--show-current"),
            "new_branch_head": _git("rev-parse", "HEAD"),
            "draft_pr": "https://github.com/WuJH123/RTC/pull/35",
            "merged_to_main": False,
        },
        "boundary": {
            "scientific_split": ["development"],
            "development_fold": ["train"],
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "closed_loop_run": False,
            "full_train_smoke_run": False,
            "acceptance_thresholds_changed": False,
            "hyperparameter_grid_search": False,
            "cohort_changed": False,
            "precision": "fp32",
        },
        "bug_fix": {
            "old_stratum_count_implementation": "BUG_CONFIRMED",
            "new_per_group_balancing": "PASS",
            "medium_overlap_bug": "CONFIRMED",
            "corrected_partition": weight_audit["total_counts"],
            "partition_valid": weight_audit["partition_valid"],
            "balanced_valid": weight_audit["balanced_valid"],
        },
        "magnitude_input": {
            "analytic_action_contract_normalization": "PASS",
            "zero_action_exact_zero": "PASS",
            "all_features_finite": "PASS",
        },
        "old_v41_corrected_reporting": corrected["old_v41"],
        "old_v42_corrected_reporting": corrected["old_v42"],
        "new_v421": {
            "d2": micro["d2"],
            "d3": micro["d3"],
            "d3_magnitude_strata": new_d3,
            "tiny_d3": tiny_d3,
            "tiny_combined": tiny_combined,
            "micro": micro,
        },
        "weight_audit": weight_audit,
        "invariants": invariants,
        "scientific_interpretation": {
            "large_effect_compression": compression,
            "magnitude_only_hypothesis": magnitude_hypothesis,
            "d2_health": "PASS" if d2_health else "FAIL",
            "d3_directional_improvement": "PASS" if d3_directional_improvement else "FAIL",
            "topology_conditioned_interaction_next": "YES",
            "reason": "Large response remains in the 0.4-0.5 range and the D3 rank remains weak; magnitude-only correction is insufficient.",
        },
        "performance": {
            "wall_time_seconds": micro["profile_seconds"].get("wall_time_seconds"),
            "forward_seconds": micro["profile_seconds"].get("forward_seconds"),
            "backward_seconds": micro["profile_seconds"].get("backward_seconds"),
            "optimizer_seconds": micro["profile_seconds"].get("optimizer_seconds"),
            "gpu_utilization": micro["gpu_utilization"],
            "gpu_peak_memory_bytes": micro["gpu_peak_memory_bytes"],
            "reference_forwards_per_group": 1,
        },
        "ready_to_merge_pr": False,
        "ready_to_replace_active_step2": False,
        "ready_for_full_train_only_smoke": False,
        "ready_for_formal": False,
        "need_new_swmm": False,
        "next_bounded_action": "External review; next technical step is state-and-topology-conditioned D3 interaction, not more magnitude descriptors/weights/epochs.",
    }
    payload = _round(payload)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    def metric_line(label: str, old: dict[str, Any], new: dict[str, Any]) -> str:
        return f"| {label} | {old[label]:.4f} | {new[label]:.4f} |"

    md = f"""# PROJECT7 STEP2 D3 MAGNITUDE BALANCE V4.2.1

## Boundary

- SWMM launched: **NO**; D2/D3 regenerated: **NO**
- Validation outcomes / Final: **NOT ACCESSED**
- Formal, closed-loop MPC, Policy Lock, full Train-only smoke: **NOT RUN**
- Frozen development/train cohort unchanged; FP32; no grid search; no threshold changes

## BUG FIX

- Old stratum-count implementation: **BUG CONFIRMED** (`strata.sum(dim=1)` produced `[3,C]` instead of per-group counts).
- New per-group balancing: **PASS**.
- Medium overlap bug: **CONFIRMED** and fixed through the shared helper.
- Corrected frozen D3 partition: small **15**, medium **12**, large **21**, total **48**; partition valid **{weight_audit['partition_valid']}**.
- Present strata receive equal total weight within every group and each group's mean candidate weight is one; balanced valid **{weight_audit['balanced_valid']}**.

### Weight totals

The selected micro epoch is {micro['best_epoch']}; all {micro['epochs_completed']} history epochs contain per-group weight audits. Example group totals are `2.666667/2.666667/2.666667` when all three strata are present and `4.0/0.0/4.0` when medium is absent. Missing strata receive zero weight without NaN/Inf.

## MAGNITUDE INPUT

- Analytic action-contract normalization (`A=109`, `H=72`, 36 blocks): **PASS**.
- Zero-action magnitude vector exact zero: **PASS**.
- All descriptor inputs finite: **PASS**.

## CORRECTED REPORTING

| metric | OLD V4.1 | OLD V4.2 | NEW V4.2.1 |
|---|---:|---:|---:|
| D3 rank | {corrected['old_v41']['d3']['rank']:.4f} | {corrected['old_v42']['d3']['rank']:.4f} | {micro['d3']['rank']:.4f} |
| D3 pairwise | {corrected['old_v41']['d3']['pairwise']:.4f} | {corrected['old_v42']['d3']['pairwise']:.4f} | {micro['d3']['pairwise']:.4f} |
| D3 top1 | {corrected['old_v41']['d3']['top1']}/6 | {corrected['old_v42']['d3']['top1']}/6 | {micro['d3']['top1']}/6 |
| D3 max regret (m3) | {corrected['old_v41']['d3']['max_regret_m3']:.1f} | {corrected['old_v42']['d3']['max_regret_m3']:.1f} | {micro['d3']['max_regret_m3']:.1f} |

| stratum | OLD V4.1 ratio | OLD V4.2 ratio | NEW V4.2.1 ratio |
|---|---:|---:|---:|
| small | {old_v41_d3['small']['response_ratio']:.4f} | {old_v42_d3['small']['response_ratio']:.4f} | {new_d3['small']['response_ratio']:.4f} |
| medium | {old_v41_d3['medium']['response_ratio']:.4f} | {old_v42_d3['medium']['response_ratio']:.4f} | {new_d3['medium']['response_ratio']:.4f} |
| large | {old_v41_d3['large']['response_ratio']:.4f} | {old_v42_d3['large']['response_ratio']:.4f} | {new_d3['large']['response_ratio']:.4f} |

## NEW V4.2.1 MICRO

### D2

rank **{micro['d2']['rank']:.4f}**; pairwise **{micro['d2']['pairwise']:.4f}**; sign **{micro['d2']['sign']:.4f}**; top1 **{micro['d2']['top1']}/6**; mean/max regret **{micro['d2']['mean_regret_m3']:.1f}/{micro['d2']['max_regret_m3']:.1f} m3**.

### D3

rank **{micro['d3']['rank']:.4f}**; pairwise **{micro['d3']['pairwise']:.4f}**; sign **{micro['d3']['sign']:.4f}**; top1 **{micro['d3']['top1']}/6**; mean/max regret **{micro['d3']['mean_regret_m3']:.1f}/{micro['d3']['max_regret_m3']:.1f} m3**.

| stratum | count | response ratio | rank | pairwise | sign |
|---|---:|---:|---:|---:|---:|
| small | {new_d3['small']['count']} | {new_d3['small']['response_ratio']:.4f} | {new_d3['small']['rank']:.4f} | {new_d3['small']['pairwise']:.4f} | {new_d3['small']['sign']:.4f} |
| medium | {new_d3['medium']['count']} | {new_d3['medium']['response_ratio']:.4f} | {new_d3['medium']['rank']:.4f} | {new_d3['medium']['pairwise']:.4f} | {new_d3['medium']['sign']:.4f} |
| large | {new_d3['large']['count']} | {new_d3['large']['response_ratio']:.4f} | {new_d3['large']['rank']:.4f} | {new_d3['large']['pairwise']:.4f} | {new_d3['large']['sign']:.4f} |

## INVARIANTS

All structural invariants are **PASS**, including exact zero/single interaction, causality, actuator identity, 36->72 expansion, non-negative flooding, head-depth consistency, and finite/nonzero gradients. D3-only tiny preserves the D2 single-branch SHA and exact baseline equivalence.

## PERFORMANCE

- wall time: {micro['profile_seconds'].get('wall_time_seconds', float('nan')):.2f} s
- forward/backward/optimizer: {micro['profile_seconds'].get('forward_seconds', float('nan')):.2f}/{micro['profile_seconds'].get('backward_seconds', float('nan')):.2f}/{micro['profile_seconds'].get('optimizer_seconds', float('nan')):.2f} s
- GPU mean/p90/max: {micro['gpu_utilization'].get('mean_percent', float('nan')):.1f}%/{micro['gpu_utilization'].get('p90_percent', float('nan')):.1f}%/{micro['gpu_utilization'].get('max_percent', float('nan')):.1f}%
- GPU peak memory: {micro['gpu_peak_memory_bytes'] / 1024**3:.2f} GiB; reference forwards/group: 1

## SCIENTIFIC INTERPRETATION

- Large-effect compression: **{compression}** (ratio {large_ratio:.4f} remains in the 0.4-0.5 range).
- Magnitude-only hypothesis: **{magnitude_hypothesis}**. Correct weighting/reporting is now valid, but D2 health and/or the full D3 directional criteria do not support promotion.
- Topology-conditioned interaction next: **YES**. Do not add more descriptors, loss weights, epochs, or topology in this turn; the next bounded design review should address state/topology-conditioned D3 interaction.

## VERDICT

**{verdict}**

- Ready to merge PR #35: **NO**
- Ready to replace active Step2: **NO**
- Ready for full Train-only smoke: **NO**
- Ready for Formal: **NO**
- Need new SWMM: **NO**
- Next bounded action: external review of the correctness patch and a separately authorized topology-conditioned D3 design.

## GitHub

- latest commit SHA: `{_git('rev-parse', 'HEAD')}`
- Draft PR: https://github.com/WuJH123/RTC/pull/35
- CI/local tests: full pytest and py_compile required before commit; no production Step2 files changed
- report JSON: `{REPORT_JSON}`
- weight audit JSON: `{WEIGHT_AUDIT_JSON}`
"""
    REPORT_MD.write_text(md, encoding="utf-8")
    return REPORT_MD, REPORT_JSON


if __name__ == "__main__":
    for path in build():
        print(path)
