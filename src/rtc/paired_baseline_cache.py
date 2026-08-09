from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .baseline_cache_v3 import build_baseline_cache
from .baselines import canonical_baseline_id
from .inp_runtime import build_runtime_inp, sha256_file


def _paired_internal_registry(
    event_registry: pd.DataFrame,
    *,
    output_dir: Path,
    native_controls_template: str | Path,
    swmm_threads_per_process: int,
) -> pd.DataFrame:
    template = Path(native_controls_template).resolve()
    if not template.is_file():
        raise ValueError(f"native-controls template missing: {template}")
    paired_dir = output_dir / "_paired_internal_event_sources"
    paired_dir.mkdir(parents=True, exist_ok=True)
    frame = event_registry.copy()
    paths: list[str] = []
    for _, row in frame.iterrows():
        source = Path(str(row["inp_path"])).resolve()
        if not source.is_file():
            raise ValueError(f"event INP missing: {source}")
        event_id = str(row["event_id"])
        paired = paired_dir / (
            f"{event_id}__{sha256_file(source)[:12]}__"
            f"rules_{sha256_file(template)[:12]}.inp"
        )
        if not paired.is_file():
            build_runtime_inp(
                source,
                paired,
                native_controls=True,
                swmm_threads=swmm_threads_per_process,
                native_controls_template=template,
            )
        paths.append(str(paired.resolve()))
    frame["inp_path"] = paths
    return frame


def build_event_paired_baseline_cache(
    *,
    event_registry: pd.DataFrame,
    output_dir: str | Path,
    config_path: str | Path,
    strategies: Iterable[str],
    stage: str,
    workers: int,
    swmm_threads_per_process: int,
    native_controls_template: str | Path,
    force: bool = False,
    formalize_final: bool = False,
) -> pd.DataFrame:
    """Generate fixed baselines while pairing Internal-RTC rules with exact event forcing.

    All strategies use the same event-specific DWF/rainfall/initial-condition source. Only the
    Internal-RTC registry view receives a generated source whose ``[CONTROLS]`` body is copied
    from the frozen native-controls template. Scientific event identity intentionally ignores
    ``[CONTROLS]``, so paired Formal event hashes remain comparable to the locked event registry.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized = tuple(canonical_baseline_id(x) for x in strategies)
    frames: list[pd.DataFrame] = []
    for strategy in normalized:
        registry = event_registry
        if strategy == "internal_rtc":
            registry = _paired_internal_registry(
                event_registry,
                output_dir=out,
                native_controls_template=native_controls_template,
                swmm_threads_per_process=swmm_threads_per_process,
            )
        frame = build_baseline_cache(
            event_registry=registry,
            output_dir=out,
            config_path=config_path,
            strategies=(strategy,),
            stage=stage,
            workers=workers,
            swmm_threads_per_process=swmm_threads_per_process,
            force=force,
            formalize_final=formalize_final,
        )
        frames.append(frame)
    if not frames:
        raise ValueError("no fixed baseline strategies requested")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["native_controls_template_sha256"] = None
    mask = combined["strategy"].astype(str) == "internal_rtc"
    combined.loc[mask, "native_controls_template_sha256"] = sha256_file(
        native_controls_template
    )
    return combined.sort_values(["event_id", "strategy"]).reset_index(drop=True)
