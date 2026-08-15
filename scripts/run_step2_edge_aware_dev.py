"""Run the edge-aware V128 ablation through the current smoke/dev staged pipeline.

This wrapper is intentionally not a current/full entrypoint. It reuses the same deterministic
profile runner and training objectives, replacing only the hydraulic transition representation.
Resume stays disabled because edge-artifact SHA is not a top-level stage-lineage field; this
prevents a caller from supplying a different artifact that would be overwritten by checkpoint
buffers during state restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import run_step2_v128_current_profiles as runner
from rtc.edge_physics_current_v128 import load_edge_physics_artifact_v128
from rtc.step2_differentiable_v128_edge import V128_EDGE_AWARE_CONTRACT, build_v128_edge_aware_model_from_graph
from rtc.step2_lazy_stream_v128 import install_v128_lazy_streaming

WRAPPER_CONTRACT = "PROJECT7_V128_EDGE_AWARE_SMOKE_DEV_WRAPPER_V2_LAZY_STREAM"


def _extract_edge_path(argv: list[str]) -> tuple[str, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--edge-physics", required=True)
    known, remaining = p.parse_known_args(argv)
    return str(known.edge_physics), remaining


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    edge_path, remaining = _extract_edge_path(sys.argv[1:])
    if "--resume-from" in remaining:
        raise ValueError("edge-aware wrapper forbids --resume-from until edge artifact SHA enters stage lineage")
    try:
        profile_index = remaining.index("--profile")
        profile = remaining[profile_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("edge-aware wrapper requires explicit --profile smoke|dev") from exc
    if profile not in {"smoke", "dev"}:
        raise ValueError("edge-aware architecture is Development-only; --profile full is forbidden")

    install_v128_lazy_streaming()
    original_builder = runner.build_v128_model_from_graph

    def edge_builder(graph, **kwargs):
        artifact = load_edge_physics_artifact_v128(edge_path, graph)
        return build_v128_edge_aware_model_from_graph(graph, edge_artifact=artifact, **kwargs)

    runner.build_v128_model_from_graph = edge_builder
    runner.CURRENT_RUN_CONTRACT = WRAPPER_CONTRACT
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *remaining]
        runner.main()
    finally:
        sys.argv = previous_argv
        runner.build_v128_model_from_graph = original_builder

    if "--out-dir" in remaining:
        out = Path(remaining[remaining.index("--out-dir") + 1])
        report = out / runner.REPORT_FILENAME
        if report.is_file():
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["architecture"] = V128_EDGE_AWARE_CONTRACT
            payload["edge_physics_sha256"] = _sha(edge_path)
            payload["lazy_mmap_branch_streaming"] = True
            payload["scientific_claim_allowed"] = False
            payload["full_promotion_allowed"] = False
            report.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
