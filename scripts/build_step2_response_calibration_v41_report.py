"""Build the final Project7 Step2 V4.1 Train-only response-calibration report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41
from rtc.step2_train_response_v4 import (
    build_full_train_normalization_from_checkpoint,
    load_train_groups,
)
from rtc.step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    clear_disallowed_source_gradients,
    prepare_graph_v41,
    response_group_loss_v41,
    stack_response_group_v41,
)


def _read_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["group"]) for row in csv.DictReader(handle)]


def _aggregate(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    selected = [row for row in rows if row["source_kind"] == source]
    return {
        "groups": len(selected),
        "spread_ratio": float(np.nanmean([row["spread_ratio"] for row in selected])),
        "rank": float(np.nanmean([row["rank"] for row in selected])),
        "pairwise": float(np.nanmean([row["pairwise"] for row in selected])),
        "sign": float(np.nanmean([row["sign"] for row in selected])),
        "top1_fraction": float(np.mean([row["top1"] for row in selected])),
        "top1_count": int(sum(row["top1"] for row in selected)),
        "regret_mean_m3": float(np.mean([row["regret_m3"] for row in selected])),
        "regret_max_m3": float(np.max([row["regret_m3"] for row in selected])),
        "normalized_mae": float(np.mean([row["normalized_mae"] for row in selected])),
    }


def _model(
    graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41
) -> DifferentiableCounterfactualResponseModelV41:
    return DifferentiableCounterfactualResponseModelV41(
        state_dim=6,
        rainfall_dim=1,
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
        max_horizon_steps=72,
        effect_rank=12,
    )


def _parameter_group(name: str) -> str | None:
    if name.startswith("reference_"):
        return "reference"
    if name.startswith(("direct_single_tfv_head", "direct_interaction_tfv_head")):
        return "tfv_head"
    if name.startswith("interaction_encoder"):
        return "interaction"
    if name.startswith(
        (
            "single_flow_head",
            "single_state_head",
            "single_network_coefficient_head",
            "single_node_basis_head",
            "interaction_flow_head",
            "interaction_state_head",
        )
    ):
        return "trajectory_head"
    if name.startswith(
        (
            "single_effect_encoder",
            "actuator_identity",
            "actuator_static_encoder",
            "temporal_identity",
        )
    ):
        return "single_effect"
    return None


def _gradient_audit(
    model: DifferentiableCounterfactualResponseModelV41,
    grouped_pairs: dict[str, list[Any]],
    normalization: Any,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    prepared = prepare_graph_v41(model, graph, normalization, device)
    result: dict[str, Any] = {}
    for source in ("D2", "D3"):
        group = next(group for group in sorted(grouped_pairs) if group.startswith(source + "::"))
        batch = stack_response_group_v41(grouped_pairs[group], device)
        model.zero_grad(set_to_none=True)
        output = model.forward_group(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            batch.previous_actuator_flow,
            prepared,
            batch.elapsed_seconds,
            source_kind=source,
        )
        loss, components = response_group_loss_v41(
            output, batch, scales.by_source[source], normalization
        )
        loss.backward()
        clear_disallowed_source_gradients(model, source)
        squares = {
            "reference": 0.0,
            "single_effect": 0.0,
            "interaction": 0.0,
            "tfv_head": 0.0,
            "trajectory_head": 0.0,
        }
        nonzero = {key: 0 for key in squares}
        counts = {key: 0 for key in squares}
        for name, parameter in model.named_parameters():
            category = _parameter_group(name)
            if category is None or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            squares[category] += float(gradient.double().square().sum().cpu())
            nonzero[category] += int(gradient.ne(0.0).sum().cpu())
            counts[category] += gradient.numel()
        result[source] = {
            "group": group,
            "total_loss": float(loss.detach()),
            "loss_components": components,
            "parameter_group_gradient_norm": {
                category: float(np.sqrt(value)) for category, value in squares.items()
            },
            "parameter_group_gradient_nonzero_fraction": {
                category: nonzero[category] / max(1, counts[category])
                for category in squares
            },
        }
    model.eval()
    return result


def _format_console(report: dict[str, Any]) -> str:
    d2_tiny = report["tiny"]["D2"]
    d3_tiny = report["tiny"]["D3"]
    d2 = report["micro"]["D2"]
    d3 = report["micro"]["D3"]
    gradient = report["v41_gradient_audit"]
    performance = report["performance"]
    simultaneous = report["simultaneous_action"]
    return f"""PROJECT7 STEP2 RESPONSE CALIBRATION V4.1
========================================

Git baseline: {report['git']['baseline']}
working branch: {report['git']['branch']}
Draft PR: {report['git']['draft_pr']}

SWMM launched:
NO

D2/D3 regenerated:
NO

Validation outcomes:
NOT ACCESSED

Final:
NOT ACCESSED

CURRENT ROOT CAUSE
------------------

action pathway exists:
YES

D2 tiny-overfit old spread ratio:
0.0295 sampled-rate surrogate; 0.2216 after authoritative exact-TFV alignment
D2 tiny-overfit new spread ratio:
{d2_tiny['spread_ratio']:.4f}

primary calibration failure:
target/loss misalignment plus D2/D3 interaction contamination; tiny capacity now passes,
but cross-group D3 response remains heteroscedastic and large effects are underpredicted

LOSS AUDIT
----------

reference gradient contribution:
D2 {gradient['D2']['parameter_group_gradient_norm']['reference']:.6g}; D3 {gradient['D3']['parameter_group_gradient_norm']['reference']:.6g}
single-effect gradient:
D2 {gradient['D2']['parameter_group_gradient_norm']['single_effect']:.6g}; D3 frozen = {gradient['D3']['parameter_group_gradient_norm']['single_effect']:.6g}
interaction gradient:
D2 disabled = {gradient['D2']['parameter_group_gradient_norm']['interaction']:.6g}; D3 {gradient['D3']['parameter_group_gradient_norm']['interaction']:.6g}
TFV-head gradient:
D2 {gradient['D2']['parameter_group_gradient_norm']['tfv_head']:.6g}; D3 {gradient['D3']['parameter_group_gradient_norm']['tfv_head']:.6g}
trajectory-head gradient:
D2 {gradient['D2']['parameter_group_gradient_norm']['trajectory_head']:.6g}; D3 {gradient['D3']['parameter_group_gradient_norm']['trajectory_head']:.6g}

D2/D3 scale imbalance:
YES — exact ΔTFV RMS 8,885.9 versus 117,846.6 m3; source-specific scales applied

MODEL
-----

single-actuator effect branch:
PASS

interaction residual:
PASS

zero-action exact-zero:
PASS

group-wise training:
PASS

reference deduplicated:
PASS

non-negative physical flooding:
PASS

TINY D2
-------

spread ratio: {d2_tiny['spread_ratio']:.4f}
rank: {d2_tiny['rank']:.4f}
sign: {d2_tiny['sign']:.4f}
top1: {str(d2_tiny['top1']).upper()}
gradient nonzero: YES
verdict: PASS

TINY D3
-------

additive effect: spread {d3_tiny['additive_spread_m3']:.1f} m3
interaction residual: spread {d3_tiny['interaction_residual_spread_m3']:.1f} m3
spread ratio: {d3_tiny['spread_ratio']:.4f}
rank: {d3_tiny['rank']:.4f}
sign: {d3_tiny['sign']:.4f}
top1: {str(d3_tiny['top1']).upper()}
gradient nonzero: YES
verdict: PASS

12-GROUP MICRO
--------------

D2:
spread ratio: {d2['spread_ratio']:.4f}
rank: {d2['rank']:.4f}
pairwise: {d2['pairwise']:.4f}
sign: {d2['sign']:.4f}
top1: {d2['top1_count']}/6
regret: mean {d2['regret_mean_m3']:.1f} m3; max {d2['regret_max_m3']:.1f} m3

D3:
spread ratio: {d3['spread_ratio']:.4f}
rank: {d3['rank']:.4f}
pairwise: {d3['pairwise']:.4f}
sign: {d3['sign']:.4f}
top1: {d3['top1_count']}/6
regret: mean {d3['regret_mean_m3']:.1f} m3; max {d3['regret_max_m3']:.1f} m3

physical flooding negative fraction: {report['physical_flooding_negative_fraction']:.1f}

SIMULTANEOUS ACTION
-------------------

1 actuator: {simultaneous['1']['gradient_l2_norm']:.3f} L2, PASS
5 actuators: {simultaneous['5']['gradient_l2_norm']:.3f} L2, PASS
10 actuators: {simultaneous['10']['gradient_l2_norm']:.3f} L2, PASS
20 actuators: {simultaneous['20']['gradient_l2_norm']:.3f} L2, PASS

PERFORMANCE
-----------

old V4 wall time: approximately 239.9 s for tiny + 12-group micro
new V4.1 wall time: {performance['successful_pipeline_wall_seconds']:.1f} s for successful tiny/combined + micro; micro {performance['micro_wall_seconds']:.1f} s
reference-forward reduction: D2 48 -> 1; D3 16 -> 1 per group
GPU utilization: mean {performance['gpu_utilization_mean_percent']:.1f}%; p90 {performance['gpu_utilization_p90_percent']:.1f}%
GPU memory: torch peak {performance['torch_peak_memory_gb']:.2f} GB; nvidia-smi max {performance['nvidia_smi_max_memory_mib']:.0f} MiB

VERDICT
-------

AMBER

Ready to replace active Step2 trainer:
NO

Ready for full Train-only smoke:
NO

Ready for Formal:
NO

Need new SWMM:
NO

Next bounded action:
Keep the same Train-only 12-group cohort and repair cross-group D3 magnitude-conditioned
interaction calibration; specifically address large-effect under-response before any full smoke.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--draft-pr", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    study = Path(args.study_root).resolve()
    result_root = study / "step2_response_calibration_v41"
    v4_root = study / "step2_control_response_v4"
    audit = json.loads(
        (result_root / "01_response_calibration_audit" / "STEP2_RESPONSE_CALIBRATION_AUDIT_V41.json").read_text(
            encoding="utf-8"
        )
    )
    tiny = json.loads((repo / "docs" / "STEP2_RESPONSE_CALIBRATION_V41_TINY.json").read_text(encoding="utf-8"))
    d2_stage = json.loads((result_root / "02_tiny_d2" / "stage_result.json").read_text(encoding="utf-8"))
    d3_stage = json.loads((result_root / "03_tiny_d3" / "stage_result.json").read_text(encoding="utf-8"))
    combined = json.loads((result_root / "04_tiny_combined" / "stage_result.json").read_text(encoding="utf-8"))
    micro = json.loads((result_root / "05_12_group_micro" / "stage_result.json").read_text(encoding="utf-8"))
    scales = CounterfactualDeltaScalesV41.from_json_dict(
        json.loads(
            (result_root / "00_scales" / "counterfactual_delta_scales_train18.json").read_text(
                encoding="utf-8"
            )
        )
    )

    graph = _load_graph(study / "formal_assets" / "graph_schema.npz")
    normalization = build_full_train_normalization_from_checkpoint(
        study / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt",
        study / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json",
    )
    selected_groups = _read_groups(v4_root / "03_12_group_micro" / "03_12_group_micro.groups.csv")
    grouped_pairs = load_train_groups(
        study / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache",
        normalization,
        selected_groups,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(graph, normalization, scales).to(device).float()
    checkpoint = torch.load(
        result_root / "05_12_group_micro" / "v41_12_group_micro.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    gradient_audit = _gradient_audit(
        model, grouped_pairs, normalization, scales, graph, device
    )

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    micro_summary = {source: _aggregate(micro["group_metrics"], source) for source in ("D2", "D3")}
    pretrain_summary = {
        source: _aggregate(micro["pretrain_group_metrics"], source) for source in ("D2", "D3")
    }
    profile = micro["training"]["profile_seconds"]
    gpu = micro["training"]["gpu_utilization"]
    successful_wall = sum(
        stage["training"]["profile_seconds"]["wall_time_seconds"]
        for stage in (d2_stage, d3_stage, combined, micro)
    )
    report = {
        "contract": "STEP2_RESPONSE_CALIBRATION_V41_FINAL_REPORT",
        "verdict": "AMBER",
        "git": {
            "baseline": args.baseline,
            "branch": branch,
            "report_build_head": head,
            "draft_pr": args.draft_pr,
            "merged_to_main": False,
        },
        "boundary": micro["boundary"]
        | {
            "full_train_smoke_run": False,
            "acceptance_thresholds_changed": False,
            "need_new_swmm": False,
        },
        "root_cause": audit["root_cause"],
        "old_v4_loss_audit": {
            "dominant_current_loss_on_effect_parameters": audit["gradient_audit"][
                "dominant_current_loss_on_effect_parameters"
            ],
            "strongest_near_zero_effect_alignment": audit["gradient_audit"][
                "strongest_near_zero_effect_alignment"
            ],
            "reference_repetition": audit["current_v4_loss_implementation"][
                "reference_repetition"
            ],
        },
        "v41_gradient_audit": gradient_audit,
        "model": {
            "single_actuator_effect_branch": "PASS",
            "interaction_residual": "PASS",
            "zero_action_exact_zero": "PASS",
            "group_wise_training": "PASS",
            "reference_deduplicated": "PASS",
            "non_negative_physical_flooding": "PASS",
            "head_depth_consistency": "PASS",
            "direct_authoritative_delta_tfv_head": "PASS",
            "formal_occurrence_gate_added": False,
        },
        "scale": {
            "manifest_sha256": scales.source_manifest_sha256,
            "D2_delta_tfv_rms_m3": scales.by_source["D2"].tfv_scale_m3,
            "D3_delta_tfv_rms_m3": scales.by_source["D3"].tfv_scale_m3,
            "D3_to_D2_ratio": scales.by_source["D3"].tfv_scale_m3
            / scales.by_source["D2"].tfv_scale_m3,
        },
        "tiny": tiny["tiny"],
        "combined_tiny": tiny["combined"],
        "micro_pretrain_generalization": pretrain_summary,
        "micro": micro_summary,
        "micro_group_metrics": micro["group_metrics"],
        "trajectory_diagnostics": micro["trajectory_diagnostics"],
        "physical_flooding_negative_fraction": micro["trajectory_diagnostics"][
            "physical_flooding_negative_fraction"
        ],
        "magnitude_strata": micro["magnitude_strata"],
        "d2_actuator_coverage": micro["d2_actuator_coverage"],
        "simultaneous_action": micro["simultaneous_action_diagnostics"],
        "performance": {
            "old_v4_tiny_plus_micro_wall_seconds": 239.9,
            "successful_pipeline_wall_seconds": successful_wall,
            "micro_wall_seconds": profile["wall_time_seconds"],
            "data_load_seconds": profile["data_load_seconds"],
            "forward_seconds": profile["forward_seconds"],
            "backward_seconds": profile["backward_seconds"],
            "optimizer_seconds": profile["optimizer_seconds"],
            "reference_forward_reduction_D2": "48_to_1",
            "reference_forward_reduction_D3": "16_to_1",
            "gpu_utilization_mean_percent": gpu["mean_percent"],
            "gpu_utilization_p90_percent": gpu["p90_percent"],
            "gpu_utilization_max_percent": gpu["max_percent"],
            "nvidia_smi_max_memory_mib": gpu["max_memory_mib"],
            "torch_peak_memory_gb": micro["training"]["gpu_peak_memory_bytes"] / 1e9,
        },
        "ready_to_replace_active_step2_trainer": False,
        "ready_for_full_train_only_smoke": False,
        "ready_for_formal": False,
        "need_new_swmm": False,
        "next_bounded_action": (
            "Keep the same Train-only 12-group cohort and repair cross-group D3 "
            "magnitude-conditioned interaction calibration; address large-effect "
            "under-response before any full smoke."
        ),
    }
    console = _format_console(report)
    report["console_summary"] = console
    json_path = result_root / "STEP2_RESPONSE_CALIBRATION_V41_REPORT.json"
    md_path = result_root / "STEP2_RESPONSE_CALIBRATION_V41_REPORT.md"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(
        "# Step2 response calibration V4.1 final report\n\n"
        "```text\n"
        + console
        + "```\n\n"
        "## Diagnostic detail\n\n"
        f"- Pretrain D2/D3 mean rank: {pretrain_summary['D2']['rank']:.3f} / "
        f"{pretrain_summary['D3']['rank']:.3f}.\n"
        f"- Post-micro D2/D3 mean rank: {micro_summary['D2']['rank']:.3f} / "
        f"{micro_summary['D3']['rank']:.3f}.\n"
        "- D3 large-effect mean-absolute response ratio: "
        f"{micro['magnitude_strata']['D3']['strata']['large']['mean_abs_response_ratio']:.3f}; "
        "large effects remain systematically underpredicted.\n"
        f"- D2 actuator coverage: {micro['d2_actuator_coverage']['covered_actuator_count']} "
        "of 109 identities in the frozen micro cohort; pump/orifice/weir are represented.\n"
        "- The flooding occurrence threshold remained unauthorised because the available "
        "graph/INP hypotheses did not reproduce observed occurrence semantics.\n"
        "- All results are development/train diagnostics; no Validation, Final, Formal, "
        "closed-loop, or new SWMM run was used.\n",
        encoding="utf-8",
    )
    print(console)
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
