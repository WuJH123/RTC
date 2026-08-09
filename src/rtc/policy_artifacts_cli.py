from __future__ import annotations

import argparse
import json
from pathlib import Path


_CANONICAL_RELATIVE = {
    "fresh_workspace_manifest": "FRESH_WORKSPACE_MANIFEST.json",
    "inp_preflight": "preflight/inp_audit.json",
    "time_scale_config": "contracts/time_scale_config.json",
    "study_readiness": "contracts/study_readiness.json",
    "step1_model": "models/step1.pt",
    "step2_model": "models/step2.pt",
    "graph_schema": "formal_assets/graph_schema.npz",
    "split_registry": "contracts/event_registry_with_splits.csv",
    "model_acceptance_contract": "contracts/model_acceptance_contract.json",
    "step1_acceptance": "acceptance/step1_gate.json",
    "step2_acceptance": "acceptance/step2_gate.json",
    "gradient_acceptance": "acceptance/gradient_gate.json",
    "candidate_ranking_acceptance": "acceptance/ranking.json",
    "controller_config": "contracts/controller_resolved.json",
    "runtime_acceptance": "acceptance/runtime_acceptance.json",
}


def build_policy_artifact_map(
    *,
    root: str | Path,
    frozen_inp: str | Path,
    priority_nodes: str | Path,
    sensor_layout: str | Path,
    baseline_plan: str | Path,
    output_path: str | Path,
) -> dict[str, str]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"study root does not exist: {root_path}")
    artefacts = {
        key: str((root_path / relative).resolve())
        for key, relative in _CANONICAL_RELATIVE.items()
    }
    artefacts.update(
        {
            "frozen_inp": str(Path(frozen_inp).expanduser().resolve()),
            "priority_nodes": str(Path(priority_nodes).expanduser().resolve()),
            "sensor_layout": str(Path(sensor_layout).expanduser().resolve()),
            "baseline_plan": str(Path(baseline_plan).expanduser().resolve()),
        }
    )
    missing = [f"{name}: {path}" for name, path in artefacts.items() if not Path(path).is_file()]
    if missing:
        raise ValueError(
            "cannot build Policy-Lock artifact map because required files are missing:\n"
            + "\n".join(missing)
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artefacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artefacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical v0.6.6 Policy-Lock artifact map from a completed study root"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--frozen-inp", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--baseline-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_policy_artifact_map(
        root=args.root,
        frozen_inp=args.frozen_inp,
        priority_nodes=args.priority,
        sensor_layout=args.sensors,
        baseline_plan=args.baseline_plan,
        output_path=args.out,
    )
    print(json.dumps({"contract": "POLICY_LOCK_ARTIFACT_MAP_V2_READINESS_BOUND", "artifacts": payload}, indent=2))


if __name__ == "__main__":
    main()
