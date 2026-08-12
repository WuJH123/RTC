"""Run the V5.0 Train-only action identifiability/candidate-manifold audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtc.production_cli import _load_graph  # noqa: E402
from rtc.step2_identifiability_v50 import scan_train_action_identifiability_v50  # noqa: E402


ROOT = Path(r"E:\RTC_sewer\Project7")
REPO = ROOT / "repo"
STUDY = ROOT / "study_v069"
CACHE = STUDY / "step2_counterfactual_stability_v2" / "step2_training_cache_v1" / "CACHE_MANIFEST.json"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
DOCS = REPO / "docs"


def _markdown(report: dict) -> str:
    d3 = report["source"].get("D3", {})
    manifold = report["mpc_candidate_manifold"]
    lines = [
        "# STEP2 ACTION IDENTIFIABILITY AUDIT V5.0",
        "",
        "Train-only read-only census. SWMM, Validation and Final were not accessed.",
        "",
        f"- D3 groups: **{report['d3_group_count']}**",
        f"- D3 candidates: **{report['d3_candidate_count']}**",
        f"- Identifiability: **{report['current_d3_identifiability']}**",
        f"- D3_V2 authorized: **{report['d3_v2_authorized']}**",
        "",
        "## Action rank",
        "",
        f"- Median per-group rank-95: {d3.get('group_effective_rank_95', {}).get('median')}",
        f"- Global rank-90/95/99: {d3.get('global_action_rank', {}).get('rank_90')} / {d3.get('global_action_rank', {}).get('rank_95')} / {d3.get('global_action_rank', {}).get('rank_99')}",
        f"- Global feature dimension: {d3.get('global_action_rank', {}).get('features')}",
        "",
        "## Coverage and disentanglement",
        "",
        f"- Continuous MPC support out-of-support fraction: {manifold.get('out_of_support_fraction')}",
        f"- Action nearest-neighbour median distance: {manifold.get('nearest_neighbor_distance', {}).get('median')}",
        f"- Temporal rank-95: {d3.get('temporal_effective_rank', {}).get('rank_95')}",
        f"- Never-cochanged actuator pairs: {d3.get('coaction', {}).get('pairs_never_jointly_changed')}",
        "",
        "## Decision",
        "",
        "The production-intended MPC support is continuous all-actuator projected-gradient control; it has no finite candidate cap. The Train cache therefore receives a support/rate check plus an observed-density proxy rather than an invented finite pack.",
        "",
        "```json",
        json.dumps(report, indent=2, allow_nan=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    graph = _load_graph(str(GRAPH))
    report = scan_train_action_identifiability_v50(
        CACHE,
        actuator_physics=graph.actuator_physics,
        actuator_physics_feature_names=tuple(graph.actuator_physics_feature_names),
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "STEP2_ACTION_IDENTIFIABILITY_AUDIT_V50.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (DOCS / "STEP2_ACTION_IDENTIFIABILITY_AUDIT_V50.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    manifold = {
        "contract": report["mpc_candidate_manifold"].get("contract"),
        "boundary": report["boundary"],
        "manifold": report["mpc_candidate_manifold"],
        "source_cache_manifest_sha256": report["cache_source_manifest_sha256"],
        "d3_identifiability": report["current_d3_identifiability"],
    }
    (DOCS / "STEP2_MPC_CANDIDATE_MANIFOLD_V50.json").write_text(
        json.dumps(manifold, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (DOCS / "STEP2_MPC_CANDIDATE_MANIFOLD_V50.md").write_text(
        "# STEP2 MPC CANDIDATE MANIFOLD V5.0\n\n```json\n"
        + json.dumps(manifold, indent=2, allow_nan=True)
        + "\n```\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, allow_nan=True))
