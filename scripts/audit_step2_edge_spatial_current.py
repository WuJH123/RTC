"""Run the held-out D2 spatial audit for an edge-aware smoke/dev stage checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

import audit_step2_spatial_current as audit
from rtc.edge_physics_current_v128 import load_edge_physics_artifact_v128
from rtc.production_cli import _load_graph
from rtc.step2_differentiable_v128_edge import build_v128_edge_aware_model_from_graph


def _extract(argv: list[str]) -> tuple[str, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--edge-physics", required=True)
    known, remaining = p.parse_known_args(argv)
    return str(known.edge_physics), remaining


def main() -> None:
    edge_path, remaining = _extract(sys.argv[1:])
    try:
        checkpoint = Path(remaining[remaining.index("--stage-checkpoint") + 1])
        graph_path = remaining[remaining.index("--graph") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("edge spatial audit requires --stage-checkpoint and --graph") from exc
    graph = _load_graph(graph_path)
    artifact = load_edge_physics_artifact_v128(edge_path, graph)
    payload = torch.load(checkpoint, map_location="cpu")
    state = dict(payload.get("model_state_dict") or {})
    saved_edge = state.get("transition.edge_static_features")
    saved_length = state.get("transition.edge_effective_length_m")
    if saved_edge is None or saved_length is None:
        raise ValueError("stage checkpoint is not an edge-aware V128 checkpoint")
    expected_edge = torch.as_tensor(artifact.edge_static_features, dtype=saved_edge.dtype)
    expected_length = torch.as_tensor(artifact.effective_length_m, dtype=saved_length.dtype)
    if not torch.equal(saved_edge.cpu(), expected_edge) or not torch.equal(saved_length.cpu(), expected_length):
        raise ValueError("edge-physics artifact differs from the artifact embedded in stage checkpoint")

    original_builder = audit.build_v128_model_from_graph

    def builder(g, **kwargs):
        return build_v128_edge_aware_model_from_graph(g, edge_artifact=artifact, **kwargs)

    audit.build_v128_model_from_graph = builder
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        audit.main()
    finally:
        sys.argv = previous
        audit.build_v128_model_from_graph = original_builder


if __name__ == "__main__":
    main()
