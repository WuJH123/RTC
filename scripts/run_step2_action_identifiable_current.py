"""Current Project7 counterfactual-first Step2 Development orchestration.

The smoke/dev candidate separates direct setting response from later hydraulic feedback.
Stage A is a three-part causal curriculum:

A0 direct same-prefix setting -> managed-flow warm-up;
A1 authoritative managed-flow -> next-state hydraulic transition pretraining with direct-pair
   setting bypass blocked;
A2 joint predicted-flow teacher forcing with direct same-prefix flow/state effects.

B0 is the explicit-lazy autoregressive trajectory stage where full network feedback belongs. The
exact H360 TFV objective remains downstream and is followed only by a low-LR trajectory anchor.
TFV action gradients are audited downstream and are not SWMM training labels. ``--profile full``
remains blocked until explicit Development promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import run_step2_v128_current_profiles as runner
from rtc.edge_physics_current_v128 import load_edge_physics_artifact_v128
from rtc.step2_counterfactual_first_v128 import (
    COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
    COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
    DIRECT_ACTION_FLOW_SCALE_CONTRACT,
    build_counterfactual_first_v128_model_from_graph,
    derive_direct_response_scales_v128,
)
from rtc.step2_counterfactual_training_v5 import (
    COUNTERFACTUAL_B0_V5_CONTRACT,
    COUNTERFACTUAL_STAGE_A_V5_CONTRACT,
    DIRECT_FLOW_A0_V5_CONTRACT,
    JOINT_DIRECT_A2_V5_CONTRACT,
    ORACLE_HYDRAULIC_A1_V5_CONTRACT,
    POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
    train_counterfactual_first_stage_a_v5,
    train_counterfactual_objective_stage_v5,
    train_counterfactual_rollout_b0_v5,
)
from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming
from rtc.step2_oracle_isolation_v128 import ORACLE_FLOW_ISOLATION_CONTRACT

CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT = (
    "PROJECT7_V128_CURRENT_COUNTERFACTUAL_FIRST_EDGE_PHYSICS_SMOKE_DEV_V6_STAGE_A_B0_V5"
)


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _enhanced_source_sha256() -> str:
    digest = hashlib.sha256()
    for module_name in (
        "rtc.step2_counterfactual_first_v128",
        "rtc.step2_counterfactual_training_v5",
        "rtc.step2_oracle_isolation_v128",
        "rtc.step2_action_identifiable_v128",
        "rtc.step2_differentiable_v128_edge",
        "rtc.edge_physics_current_v128",
    ):
        module = importlib.import_module(module_name)
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise RuntimeError(f"cannot fingerprint enhanced source module: {module_name}")
        path = Path(raw)
        digest.update(module_name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _delegate_help() -> None:
    print(
        "Current counterfactual-first Step2 requires --edge-physics <EDGE_PHYSICS.npz>. "
        "Only --profile smoke|dev is enabled while this candidate is under Development gates; "
        "--profile full fails closed until explicit promotion.\n\n"
        "Delegated current profile options follow:\n",
        flush=True,
    )
    previous = sys.argv
    try:
        sys.argv = [previous[0], "--help"]
        runner.main()
    finally:
        sys.argv = previous


def _extract_edge_path(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--edge-physics", required=True)
    known, remaining = parser.parse_known_args(argv)
    return str(known.edge_physics), remaining


def _requested_profile(argv: list[str]) -> str:
    try:
        index = argv.index("--profile")
        return str(argv[index + 1]).lower()
    except (ValueError, IndexError) as exc:
        raise ValueError("current Step2 requires explicit --profile smoke|dev") from exc


def main() -> None:
    argv = list(sys.argv[1:])
    if any(value in {"-h", "--help"} for value in argv):
        _delegate_help()
        return

    edge_path, remaining = _extract_edge_path(argv)
    profile = _requested_profile(remaining)
    if profile not in {"smoke", "dev"}:
        raise ValueError(
            "current counterfactual-first V128 is Development-only. --profile full is blocked "
            "until held-out direct-flow/direct-hydraulic/trajectory/ranking/gradient and "
            "closed-loop Development gates pass."
        )
    edge_sha = _sha(edge_path)
    source_sha = _enhanced_source_sha256()
    # Exact H360 training still owns imported streaming globals. Install its audited lazy helpers;
    # Stage A/B0/post-objective V5 call lazy helpers directly and do not depend on this mutation.
    install_v128_lazy_streaming()

    original_builder = runner.build_v128_model_from_graph
    original_scale = runner.derive_residual_scales_streaming_v127
    original_stage_a = runner.train_hydraulic_stage_streaming_v128
    original_b0 = runner.train_truncated_rollout_stage_streaming_v127
    original_objective = runner.train_objective_stage_streaming_v128
    original_save_stage = runner.save_stage_checkpoint_v128
    original_load_stage = runner.load_stage_checkpoint_v128
    original_contract = runner.CURRENT_RUN_CONTRACT

    scale_context: dict[str, object] = {}

    def enhanced_scale(caches_and_names, *, sample_rows=131_072):
        state, temporal, direct, telemetry = derive_direct_response_scales_v128(
            caches_and_names, sample_rows=sample_rows
        )
        scale_context["direct_action_flow_scale"] = direct
        return state, temporal, telemetry

    def enhanced_builder(graph, **kwargs):
        artifact = load_edge_physics_artifact_v128(edge_path, graph)
        direct = scale_context.get("direct_action_flow_scale")
        if direct is None:
            raise RuntimeError("direct action-flow scale was not derived before model construction")
        return build_counterfactual_first_v128_model_from_graph(
            graph,
            edge_artifact=artifact,
            direct_action_flow_scale=direct,
            **kwargs,
        )

    def _lineage_fields() -> dict[str, object]:
        return {
            "edge_physics_sha256": edge_sha,
            "action_identifiable_source_sha256": source_sha,
            "counterfactual_first_model_contract": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
            "counterfactual_first_training_contract": COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
            "direct_action_flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
            "oracle_flow_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
            "counterfactual_stage_a_contract": COUNTERFACTUAL_STAGE_A_V5_CONTRACT,
            "direct_flow_a0_contract": DIRECT_FLOW_A0_V5_CONTRACT,
            "oracle_hydraulic_a1_contract": ORACLE_HYDRAULIC_A1_V5_CONTRACT,
            "joint_direct_a2_contract": JOINT_DIRECT_A2_V5_CONTRACT,
            "counterfactual_b0_contract": COUNTERFACTUAL_B0_V5_CONTRACT,
            "post_objective_trajectory_anchor_contract": POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
            "gradient_is_training_target": False,
            "explicit_lazy_stage_a_b0_anchor": True,
        }

    def enhanced_save_stage(path, **kwargs):
        lineage = dict(kwargs["lineage"])
        lineage.update(_lineage_fields())
        kwargs["lineage"] = lineage
        return original_save_stage(path, **kwargs)

    def enhanced_load_stage(path, **kwargs):
        lineage = dict(kwargs["expected_lineage"])
        lineage.update(_lineage_fields())
        kwargs["expected_lineage"] = lineage
        return original_load_stage(path, **kwargs)

    runner.build_v128_model_from_graph = enhanced_builder
    runner.derive_residual_scales_streaming_v127 = enhanced_scale
    runner.train_hydraulic_stage_streaming_v128 = train_counterfactual_first_stage_a_v5
    runner.train_truncated_rollout_stage_streaming_v127 = train_counterfactual_rollout_b0_v5
    runner.train_objective_stage_streaming_v128 = train_counterfactual_objective_stage_v5
    runner.save_stage_checkpoint_v128 = enhanced_save_stage
    runner.load_stage_checkpoint_v128 = enhanced_load_stage
    runner.CURRENT_RUN_CONTRACT = CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT

    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *remaining]
        runner.main()
    finally:
        sys.argv = previous_argv
        runner.build_v128_model_from_graph = original_builder
        runner.derive_residual_scales_streaming_v127 = original_scale
        runner.train_hydraulic_stage_streaming_v128 = original_stage_a
        runner.train_truncated_rollout_stage_streaming_v127 = original_b0
        runner.train_objective_stage_streaming_v128 = original_objective
        runner.save_stage_checkpoint_v128 = original_save_stage
        runner.load_stage_checkpoint_v128 = original_load_stage
        runner.CURRENT_RUN_CONTRACT = original_contract

    if "--out-dir" in remaining:
        out = Path(remaining[remaining.index("--out-dir") + 1])
        report = out / runner.REPORT_FILENAME
        if report.is_file():
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload.update(
                {
                    "architecture": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
                    "training_amendment": COUNTERFACTUAL_FIRST_TRAINING_CONTRACT,
                    "counterfactual_stage_a_contract": COUNTERFACTUAL_STAGE_A_V5_CONTRACT,
                    "direct_action_flow_scale_contract": DIRECT_ACTION_FLOW_SCALE_CONTRACT,
                    "oracle_flow_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
                    "direct_flow_a0_contract": DIRECT_FLOW_A0_V5_CONTRACT,
                    "oracle_hydraulic_a1_contract": ORACLE_HYDRAULIC_A1_V5_CONTRACT,
                    "joint_direct_a2_contract": JOINT_DIRECT_A2_V5_CONTRACT,
                    "counterfactual_b0_contract": COUNTERFACTUAL_B0_V5_CONTRACT,
                    "post_objective_trajectory_anchor_contract": POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
                    "edge_physics_sha256": edge_sha,
                    "action_identifiable_source_sha256": source_sha,
                    "explicit_lazy_stage_a_b0_anchor": True,
                    "gradient_is_training_target": False,
                    "gradient_role": "downstream diagnostic and online differentiable solver signal",
                    "development_only": True,
                    "scientific_claim_allowed": False,
                    "final_checkpoint_allowed": False,
                    "policy_lock_allowed": False,
                    "full_promotion_allowed": False,
                }
            )
            report.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        pause = out / "PAUSED_RUN.json"
        if pause.is_file():
            payload = json.loads(pause.read_text(encoding="utf-8"))
            payload.update(
                {
                    "architecture": COUNTERFACTUAL_FIRST_MODEL_CONTRACT,
                    "counterfactual_stage_a_contract": COUNTERFACTUAL_STAGE_A_V5_CONTRACT,
                    "oracle_flow_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
                    "edge_physics_sha256": edge_sha,
                    "explicit_lazy_stage_a_b0_anchor": True,
                    "gradient_is_training_target": False,
                    "scientific_claim_allowed": False,
                }
            )
            pause.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
