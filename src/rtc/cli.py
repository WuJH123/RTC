from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .contracts import load_priority_nodes, require_nodes_exist
from .data_design import design_independent_actuator_probes, summarise_probe_design
from .inp import discover_actuators, discover_nodes


def audit_inp_main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wuhan SWMM actuator/state contract")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    catalog = discover_actuators(args.inp)
    nodes = discover_nodes(args.inp)
    priority = load_priority_nodes(args.priority)
    require_nodes_exist(priority, nodes)
    result = {
        "inp": str(Path(args.inp).resolve()),
        "node_count": len(nodes),
        "actuator_count": len(catalog.actuators),
        "actuator_types": dict(Counter(a.kind for a in catalog.actuators)),
        "continuous_actuators": sum(a.continuous for a in catalog.actuators),
        "hard_binary_actuators": 0,
        "priority_nodes": list(priority),
        "priority_nodes_present": True,
        "fixed_active_subset": None,
        "actuator_ids": list(catalog.ids),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)


def design_probes_main() -> None:
    parser = argparse.ArgumentParser(
        description="Design same-checkpoint, single-actuator D2 counterfactual probes"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True, help="CSV with checkpoint_id and setting:<id>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()

    catalog = discover_actuators(args.inp)
    checkpoints = pd.read_csv(args.checkpoints)
    manifest = design_independent_actuator_probes(
        checkpoints,
        catalog,
        epsilon=args.epsilon,
        include_center=not args.no_center,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = summarise_probe_design(manifest)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_probes_main() -> None:
    """Execute a probe manifest locally with PySWMM.

    Candidate SHA duplicates (typically the shared centre action) are run once. Every
    branch creates a fresh Simulation and independently replays the native prefix.
    """

    from .swmm_data import run_independent_control_branch

    parser = argparse.ArgumentParser(description="Run authoritative D2 probe branches")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inp", help="default event INP; may be overridden by manifest inp_path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"candidate_action_sha256", "candidate_settings_json", "checkpoint_minutes"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    # Same action from the same event/checkpoint is one physical branch.
    dedup_cols = ["candidate_action_sha256", "checkpoint_minutes"]
    for optional in ("event_id", "rainfall_group", "inp_path"):
        if optional in manifest.columns:
            dedup_cols.append(optional)
    jobs = manifest.drop_duplicates(dedup_cols).reset_index(drop=True)
    if args.limit is not None:
        jobs = jobs.head(args.limit)

    results: list[dict[str, object]] = []
    for _, row in jobs.iterrows():
        inp = row.get("inp_path") if "inp_path" in jobs.columns else None
        if pd.isna(inp) or not inp:
            inp = args.inp
        if not inp:
            raise ValueError("an INP is required via --inp or manifest inp_path")
        branch_id = str(row["candidate_action_sha256"])[:16]
        event = str(row.get("event_id", "event"))
        checkpoint = int(row["checkpoint_minutes"])
        branch_id = f"{event}__t{checkpoint:04d}__{branch_id}"
        settings = json.loads(str(row["candidate_settings_json"]))
        result = run_independent_control_branch(
            inp_path=inp,
            checkpoint_minutes=checkpoint,
            horizon_minutes=args.horizon_minutes,
            candidate_settings=settings,
            output_dir=args.out_dir,
            branch_id=branch_id,
            python_intervention_seconds=args.stride_seconds,
        )
        results.append(
            {
                "branch_id": result.branch_id,
                "metadata_path": result.metadata_path,
                "flow_routing_error_pct": result.flow_routing_error_pct,
            }
        )
    summary_path = Path(args.out_dir) / "RUN_SUMMARY.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(json.dumps({"branches": len(results), "summary": str(summary_path)}, indent=2))
