from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .generation_contract import generation_key
from .inp_runtime import sha256_file
from .rainfall_design import validate_formal_rainfall_design


WORKSPACE_CONTRACT = "RTC_FRESH_WORKSPACE_V2_LINEAGE_NOT_PATH_BOUND"


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def initialize_fresh_workspace(
    *,
    root: str | Path,
    frozen_inp: str | Path,
    priority_nodes: str | Path,
    event_registry: str | Path,
) -> dict[str, object]:
    """Create an initially clean study root and bind the scientific input identities."""

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
        "reuse_rule": (
            "RTC-derived artefacts may live on any local volume. Reuse is decided by scientific "
            "split/lineage, implementation-contract identity, exact numerical input/config keys and "
            "generated-artifact hashes, never by directory location or file existence alone."
        ),
        "forbidden_reuse": [
            "artifacts with incompatible scientific/data contracts",
            "Final rows in pre-lock training/acceptance",
            "branches whose declared generated-artifact hashes no longer verify",
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_fresh_workspace(path: str | Path) -> dict[str, object]:
    manifest_path = _resolve(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != WORKSPACE_CONTRACT:
        raise ValueError("not a valid RTC v0.6 study workspace manifest")
    root = _resolve(str(payload.get("output_root", "")))
    if not root.is_dir() or manifest_path.parent != root:
        raise ValueError("workspace manifest/output_root mismatch")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("workspace manifest lacks input identities")
    for name, raw in inputs.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid workspace input entry: {name}")
        p = _resolve(str(raw.get("path", "")))
        if not p.is_file() or sha256_file(p) != str(raw.get("sha256", "")):
            raise ValueError(f"workspace input disappeared/changed: {name}: {p}")
    canonical = _resolve(str(payload.get("canonical_event_registry", "")))
    require_path_inside_workspace(canonical, root)
    if not canonical.is_file():
        raise ValueError("canonical event registry is missing")
    rainfall = payload.get("rainfall_design")
    if not isinstance(rainfall, dict) or rainfall.get("required_invariants_passed") is not True:
        raise ValueError("workspace lacks a valid rainfall-group split design")
    return payload


def require_path_inside_workspace(path: str | Path, workspace_root: str | Path) -> None:
    """Optional organization helper; scientific validity does not depend on this path check."""

    candidate = _resolve(path)
    root = _resolve(workspace_root)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside study workspace: {candidate}") from exc


def _resolve_index_reference(raw: str | Path, index_path: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = index_path.parent / candidate
    return candidate.resolve()


def _verify_metadata_generation(meta_path: Path, implementation_sha: str) -> None:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"branch metadata is not a JSON object: {meta_path}")
    if payload.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError(f"branch implementation contract is incompatible: {meta_path}")
    if not payload.get("generation_key_sha256"):
        raise ValueError(f"branch lacks a generation key: {meta_path}")
    hashes = payload.get("generated_artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"branch lacks hashed generated artifacts: {meta_path}")
    for field, expected in hashes.items():
        raw = payload.get(str(field))
        if not raw:
            raise ValueError(f"branch metadata lacks artifact field {field}: {meta_path}")
        artifact = meta_path.parent / str(raw)
        if not artifact.is_file() or sha256_file(artifact) != str(expected):
            raise ValueError(f"branch generated artifact missing/changed: {artifact}")


def _verify_baseline_sidecar(
    sidecar_path: Path, *, implementation_sha: str, metadata_path: Path
) -> None:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != "FIXED_BASELINE_CACHE_V2_CODE_BOUND":
        raise ValueError(f"baseline sidecar is not current V2 evidence: {sidecar_path}")
    if payload.get("rtc_source_tree_sha256") != implementation_sha:
        raise ValueError(f"baseline implementation contract is incompatible: {sidecar_path}")
    if Path(str(payload.get("main_metadata_path", ""))).resolve() != metadata_path:
        raise ValueError("baseline sidecar/main metadata mismatch")
    if sha256_file(metadata_path) != str(payload.get("main_metadata_sha256", "")):
        raise ValueError("baseline main metadata changed after cache generation")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("baseline sidecar lacks evidence")
    for path_key, hash_key in (
        ("runtime_inp", "runtime_inp_sha256"),
        ("compact_path", "compact_sha256"),
        ("node_statistics_path", "node_statistics_sha256"),
        ("decision_log_path", "decision_log_sha256"),
    ):
        artifact = Path(str(evidence.get(path_key, "")))
        if not artifact.is_file() or sha256_file(artifact) != str(evidence.get(hash_key, "")):
            raise ValueError(f"baseline cached artifact missing/changed: {artifact}")


def validate_fresh_run_index(
    *,
    run_index_path: str | Path,
    workspace_manifest_path: str | Path,
    metadata_column: str = "metadata_path",
    reject_final: bool = True,
) -> dict[str, object]:
    """Validate branch scientific lineage without requiring one physical output directory."""

    workspace = load_fresh_workspace(workspace_manifest_path)
    root = _resolve(str(workspace["output_root"]))
    index_path = _resolve(run_index_path)
    if not index_path.is_file():
        raise ValueError(f"run index is missing: {index_path}")
    frame = pd.read_csv(index_path)
    if metadata_column not in frame.columns or frame.empty:
        raise ValueError(f"run index requires non-empty {metadata_column}")
    if reject_final:
        if "scientific_split" not in frame.columns:
            raise ValueError("training/acceptance index requires scientific_split")
        if (frame["scientific_split"].astype(str) == "final").any():
            raise ValueError("Final rows are forbidden in pre-lock training/acceptance indexes")
    _, implementation_sha = generation_key("code_probe", {})
    metadata_paths: list[str] = []
    inside_count = 0
    for _, row in frame.iterrows():
        path = _resolve_index_reference(str(row[metadata_column]), index_path)
        if not path.is_file():
            raise ValueError(f"branch metadata is missing: {path}")
        try:
            path.relative_to(root)
            inside_count += 1
        except ValueError:
            pass
        sidecar_raw = str(row.get("sidecar_path", "")).strip()
        if sidecar_raw and sidecar_raw.lower() != "nan":
            sidecar = _resolve_index_reference(sidecar_raw, index_path)
            if not sidecar.is_file():
                raise ValueError(f"baseline cache sidecar is missing: {sidecar}")
            _verify_baseline_sidecar(
                sidecar, implementation_sha=implementation_sha, metadata_path=path
            )
        else:
            _verify_metadata_generation(path, implementation_sha)
        metadata_paths.append(str(path))
    return {
        "contract": "RUN_INDEX_PROVENANCE_V3_LINEAGE_BOUND",
        "workspace_manifest_sha256": sha256_file(workspace_manifest_path),
        "run_index": str(index_path),
        "run_index_sha256": sha256_file(index_path),
        "rtc_source_tree_sha256": implementation_sha,
        "rows": int(len(frame)),
        "unique_metadata_paths": int(len(set(metadata_paths))),
        "metadata_inside_workspace": int(inside_count),
        "metadata_outside_workspace": int(len(frame) - inside_count),
        "all_rows_lineage_valid": True,
        "final_rows_rejected": bool(reject_final),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rainfall-group split and initialize a clean RTC study workspace"
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
