from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .inp_runtime import sha256_file
from .rainfall_design import validate_formal_rainfall_design


WORKSPACE_CONTRACT = "RTC_FRESH_WORKSPACE_V1_NO_HISTORICAL_OUTPUT_REUSE"


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def initialize_fresh_workspace(
    *,
    root: str | Path,
    frozen_inp: str | Path,
    priority_nodes: str | Path,
    event_registry: str | Path,
) -> dict[str, object]:
    """Create a clean output root after validating the fresh rainfall/event design.

    The physical model, observation metadata and rainfall/event definitions are scientific
    inputs. The validated event registry is copied into the fresh root and becomes the
    canonical split registry. All hydraulic trajectories, counterfactual branches, model
    checkpoints, closed-loop runs and Formal evidence must subsequently be generated there.
    """

    root_path = _resolve(root)
    manifest_path = root_path / "FRESH_WORKSPACE_MANIFEST.json"
    source_inp = _resolve(frozen_inp)
    priority = _resolve(priority_nodes)
    source_registry = _resolve(event_registry)
    for name, path in {
        "frozen_inp": source_inp,
        "priority_nodes": priority,
        "event_registry_source": source_registry,
    }.items():
        if not path.is_file():
            raise ValueError(f"fresh workspace input is missing: {name}: {path}")

    # Validate before creating the output root. A bad event design must not leave behind a
    # half-initialized directory that later looks like a valid fresh study.
    rainfall_design = validate_formal_rainfall_design(pd.read_csv(source_registry))

    if root_path.exists():
        contents = list(root_path.iterdir())
        if contents:
            raise ValueError(
                f"fresh workspace must start empty; found {len(contents)} entries in {root_path}"
            )
    else:
        root_path.mkdir(parents=True, exist_ok=False)

    contracts = root_path / "contracts"
    contracts.mkdir(parents=True, exist_ok=False)
    locked_registry = contracts / "event_registry_with_splits.csv"
    shutil.copyfile(source_registry, locked_registry)

    inputs: dict[str, dict[str, str]] = {
        "frozen_inp": {"path": str(source_inp), "sha256": sha256_file(source_inp)},
        "priority_nodes": {"path": str(priority), "sha256": sha256_file(priority)},
        "event_registry_source": {
            "path": str(source_registry),
            "sha256": sha256_file(source_registry),
        },
        "event_registry": {
            "path": str(locked_registry.resolve()),
            "sha256": sha256_file(locked_registry),
        },
    }
    if inputs["event_registry_source"]["sha256"] != inputs["event_registry"]["sha256"]:
        raise RuntimeError("fresh event-registry copy changed content")

    payload: dict[str, object] = {
        "contract": WORKSPACE_CONTRACT,
        "output_root": str(root_path),
        "canonical_event_registry": str(locked_registry.resolve()),
        "rainfall_design": rainfall_design,
        "inputs": inputs,
        "admissible_preexisting_data": [
            "frozen physical INP",
            "priority/sensor observation metadata",
            "rainfall/event forcing definitions used to build the fresh event registry",
        ],
        "forbidden_reuse": [
            "historical RTC hydraulic trajectories",
            "historical baseline outcomes",
            "historical D1/D2/D3/candidate branches",
            "historical Step1/Step2 checkpoints",
            "historical calibration/acceptance/Policy-Lock/Final evidence",
        ],
        "generation_rule": "all RTC-derived outputs for this study must be created under output_root by the current code contract",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_fresh_workspace(path: str | Path) -> dict[str, object]:
    manifest_path = _resolve(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != WORKSPACE_CONTRACT:
        raise ValueError("not a valid fresh RTC workspace manifest")
    root = _resolve(str(payload.get("output_root", "")))
    if not root.is_dir() or manifest_path.parent != root:
        raise ValueError("fresh workspace manifest/output_root mismatch")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("fresh workspace manifest lacks input identities")
    for name, raw in inputs.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid fresh workspace input entry: {name}")
        p = _resolve(str(raw.get("path", "")))
        if not p.is_file() or sha256_file(p) != str(raw.get("sha256", "")):
            raise ValueError(f"fresh workspace input disappeared/changed: {name}: {p}")
    canonical = _resolve(str(payload.get("canonical_event_registry", "")))
    require_path_inside_workspace(canonical, root)
    if not canonical.is_file():
        raise ValueError("canonical fresh event registry is missing")
    rainfall = payload.get("rainfall_design")
    if not isinstance(rainfall, dict) or int(rainfall.get("rainfall_groups", 0)) < 160:
        raise ValueError("fresh workspace lacks the validated >=160-group rainfall design")
    return payload


def require_path_inside_workspace(path: str | Path, workspace_root: str | Path) -> None:
    candidate = _resolve(path)
    root = _resolve(workspace_root)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"RTC-derived output is outside fresh workspace: {candidate}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rainfall design and initialize an empty no-historical-output-reuse Formal RTC workspace"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    payload = initialize_fresh_workspace(
        root=args.root,
        frozen_inp=args.inp,
        priority_nodes=args.priority,
        event_registry=args.events,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
