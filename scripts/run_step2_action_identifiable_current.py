"""Current Project7 action-identifiable Step2 Development orchestration.

This wrapper promotes the diagnostic-driven V128 action/hydraulic repair to the *current
smoke/dev* surface while deliberately blocking ``--profile full`` until the new model passes
the existing held-out Development ranking, gradient, spatial and closed-loop gates.

It preserves the canonical current profile runner and monkey-patches only four scientific
components for this Development candidate:

1. FIT-only hybrid temporal/action-conditioned actuator flow scales;
2. action-identifiable actuator response + frozen edge-physics transition;
3. counterfactual Stage A and B0 hydraulic-effect supervision;
4. exact H360 objective followed by a small FIT-only action-effect anchor.

Stage checkpoint/resume remains supported.  The frozen edge artifact SHA and the enhanced
architecture/training source SHA are injected into stage lineage so a changed artifact/source
cannot be silently restored into an old checkpoint.
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
from rtc.step2_action_identifiable_v128 import (
    ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
    ACTION_IDENTIFIABLE_MODEL_CONTRACT,
    ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
    build_action_identifiable_v128_model_from_graph,
    derive_action_conditioned_residual_scales_v128,
    train_action_identifiable_hydraulic_stage_v128,
    train_action_identifiable_objective_stage_v128,
    train_action_identifiable_rollout_stage_v128,
)
from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming

CURRENT_ACTION_IDENTIFIABLE_RUN_CONTRACT = (
    "PROJECT7_V128_CURRENT_ACTION_IDENTIFIABLE_EDGE_PHYSICS_SMOKE_DEV_V1"
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
        "Current action-identifiable Step2 requires --edge-physics <EDGE_PHYSICS.npz>. "
        "Only --profile smoke|dev is enabled while this repair is under Development gates; "
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
            "current action-identifiable V128 is Development-only. --profile full is blocked "
            "until held-out action-flow, gradient, ranking and closed-loop Development gates pass."
        )
    edge_sha = _sha(edge_path)
    source_sha = _enhanced_source_sha256()
    install_v128_lazy_streaming()

    original_builder = runner.build_v128_model_from_graph
    original_scale = runner.derive_residual_scales_streaming_v127
    original_stage_a = runner.train_hydraulic_stage_streaming_v128
    original_b0 = runner.train_truncated_rollout_stage_streaming_v127
    original_objective = runner.train_objective_stage_streaming_v128
    original_save_stage = runner.save_stage_checkpoint_v128
    original_load_stage = runner.load_stage_checkpoint_v128
    original_contract = runner.CURRENT_RUN_CONTRACT

    def enhanced_builder(graph, **kwargs):
        artifact = load_edge_physics_artifact_v128(edge_path, graph)
        return build_action_identifiable_v128_model_from_graph(
            graph,
            edge_artifact=artifact,
            **kwargs,
        )

    def enhanced_save_stage(path, **kwargs):
        lineage = dict(kwargs["lineage"])
        lineage.update(
            {
                "edge_physics_sha256": edge_sha,
                "action_identifiable_source_sha256": source_sha,
                "action_identifiable_model_contract": ACTION_IDENTIFIABLE_MODEL_CONTRACT,
                "action_identifiable_training_contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
                "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
            }
        )
        kwargs["lineage"] = lineage
        return original_save_stage(path, **kwargs)

    def enhanced_load_stage(path, **kwargs):
        lineage = dict(kwargs["expected_lineage"])
        lineage.update(
            {
                "edge_physics_sha256": edge_sha,
                "action_identifiable_source_sha256": source_sha,
                "action_identifiable_model_contract": ACTION_IDENTIFIABLE_MODEL_CONTRACT,
                "action_identifiable_training_contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
                "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
            }
        )
        kwargs["expected_lineage"] = lineage
        return original_load_stage(path, **kwargs)

    runner.build_v128_model_from_graph = enhanced_builder
    runner.derive_residual_scales_streaming_v127 = derive_action_conditioned_residual_scales_v128
    runner.train_hydraulic_stage_streaming_v128 = train_action_identifiable_hydraulic_stage_v128
    runner.train_truncated_rollout_stage_streaming_v127 = train_action_identifiable_rollout_stage_v128
    runner.train_objective_stage_streaming_v128 = train_action_identifiable_objective_stage_v128
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
                    "architecture": ACTION_IDENTIFIABLE_MODEL_CONTRACT,
                    "training_amendment": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
                    "flow_scale_contract": ACTION_CONDITIONED_FLOW_SCALE_CONTRACT,
                    "edge_physics_sha256": edge_sha,
                    "action_identifiable_source_sha256": source_sha,
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


if __name__ == "__main__":
    main()
