from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .baselines import canonical_baseline_id
from .causal_timing import timing_from_controller_config
from .code_contract import rtc_source_tree_sha256
from .inp_runtime import sha256_file
from .production_cli_router import run_policy_main


def _model_engine(path: str, *, name: str) -> str:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError(f"{name} checkpoint lacks model_config")
    engine = str(payload["model_config"].get("swmm_engine_version", "")).strip()
    if not engine:
        raise ValueError(f"{name} checkpoint lacks SWMM engine lineage")
    return engine


def _stamp_run_metadata(
    *,
    metadata_path: Path,
    strategy: str,
    config_path: str,
    source_inp: str,
    graph: str | None,
    step1: str | None,
    step2: str | None,
    expected_swmm_engine_version: str | None,
) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy run metadata must be a JSON object")
    actual_engine = str(payload.get("swmm_engine_version", "")).strip()
    if not actual_engine:
        raise ValueError("policy run metadata lacks SWMM engine version")
    if expected_swmm_engine_version is not None and actual_engine != expected_swmm_engine_version:
        raise RuntimeError(
            "Proposed runtime SWMM engine differs from the engine used for Step1/Step2 data: "
            f"runtime={actual_engine}, trained={expected_swmm_engine_version}"
        )
    payload["rtc_source_tree_sha256"] = rtc_source_tree_sha256()
    payload["strategy"] = canonical_baseline_id(strategy)
    payload["source_inp_sha256"] = sha256_file(source_inp)
    payload["controller_config_sha256"] = sha256_file(config_path)
    payload["expected_swmm_engine_version"] = expected_swmm_engine_version
    if graph:
        payload["graph_schema_sha256"] = sha256_file(graph)
    if step1:
        payload["step1_model_sha256"] = sha256_file(step1)
    if step2:
        payload["step2_model_sha256"] = sha256_file(step2)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--graph")
    parser.add_argument("--step1")
    parser.add_argument("--step2")
    known, _ = parser.parse_known_args()
    raw = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("controller config must be a JSON object")
    strategy = canonical_baseline_id(known.strategy)
    timing = timing_from_controller_config(raw)
    timing.validate(require_full_history_before_first_control=(strategy == "proposed"))

    expected_engine: str | None = None
    if strategy == "proposed":
        if not known.step1 or not known.step2:
            raise ValueError("Proposed requires Step1 and Step2 checkpoints")
        step1_engine = _model_engine(known.step1, name="Step1")
        step2_engine = _model_engine(known.step2, name="Step2")
        if step1_engine != step2_engine:
            raise ValueError(
                f"Step1/Step2 SWMM engine lineage differs: {step1_engine} != {step2_engine}"
            )
        expected_engine = step1_engine

    run_policy_main()
    metadata_path = Path(known.out_dir) / f"{known.run_id}.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"public policy runner completed without expected metadata: {metadata_path}"
        )
    _stamp_run_metadata(
        metadata_path=metadata_path,
        strategy=known.strategy,
        config_path=known.config,
        source_inp=known.inp,
        graph=known.graph,
        step1=known.step1,
        step2=known.step2,
        expected_swmm_engine_version=expected_engine,
    )


if __name__ == "__main__":
    main()
