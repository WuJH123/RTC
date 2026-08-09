from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inp_runtime import section_has_payload
from .units import flow_rate_to_m3s, length_to_m


def _load_meta(path: str | Path) -> tuple[dict[str, object], Path]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")), p.parent


def _source_event_inp(item: pd.Series, meta: dict[str, object]) -> str:
    """Recover the original event INP when a No-control baseline cache sidecar exists."""

    raw_sidecar = item.get("sidecar_path", "")
    sidecar_text = "" if pd.isna(raw_sidecar) else str(raw_sidecar).strip()
    if sidecar_text:
        sidecar_path = Path(sidecar_text)
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (
                sidecar.get("contract") == "FIXED_BASELINE_CACHE_V2_CODE_BOUND"
                and sidecar.get("strategy") == "no_control"
            ):
                source = Path(str(sidecar.get("source_inp", "")))
                if not source.is_file():
                    raise ValueError(
                        f"baseline cache original event INP disappeared: {source}"
                    )
                return str(source)
    runtime = Path(str(meta.get("inp_path", "")))
    if not runtime.is_file():
        raise ValueError(f"checkpoint source INP disappeared: {runtime}")
    return str(runtime)


def _assert_replayable_no_control_prefix(
    meta: dict[str, object], metadata_path: str | Path
) -> None:
    contract = str(meta.get("data_contract", ""))
    if contract == "D0_D1_COMPACT_TRAJECTORY_V3_T0_CAUSAL":
        if int(meta.get("initial_observation_elapsed_seconds", -1)) != 0:
            raise ValueError(f"D0 checkpoint source does not prove t=0 inclusion: {metadata_path}")
        if meta.get("python_actuator_writes") is not False:
            raise ValueError(
                f"D2/D3 checkpoint source contains/does not prove absence of Python writes: {metadata_path}"
            )
        if meta.get("native_controls_enabled") is not False:
            raise ValueError(
                f"D2/D3 checkpoint source has native controls enabled: {metadata_path}"
            )
        return
    if contract == "CLOSED_LOOP_COMPACT_V2":
        if meta.get("controller_present") is not False:
            raise ValueError(
                f"D2/D3 checkpoint source contains Python decisions: {metadata_path}"
            )
        inp_path = Path(str(meta.get("inp_path", "")))
        if not inp_path.is_file() or section_has_payload(inp_path, "CONTROLS"):
            raise ValueError(
                f"D2/D3 checkpoint source is not controls-disabled No-control: {metadata_path}"
            )
        decision_name = meta.get("decision_file")
        if not decision_name:
            raise ValueError("cached No-control metadata lacks decision_file lineage")
        decision_path = Path(metadata_path).parent / str(decision_name)
        if not decision_path.is_file() or decision_path.read_text(encoding="utf-8").strip():
            raise ValueError("cached No-control decision log must exist and be empty")
        return
    raise ValueError(
        f"D2/D3 checkpoint source is not a current replayable No-control trajectory: {metadata_path}"
    )


def _state_table(
    metadata_path: str | Path,
) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray, dict[str, object]]:
    meta, root = _load_meta(metadata_path)
    compact_name = meta.get("compact_file")
    if compact_name:
        with np.load(root / str(compact_name), allow_pickle=False) as raw:
            times = raw["elapsed_seconds"].astype(np.int64)
            state = raw["state_si"].astype(np.float32)
            actuator_ids = tuple(raw["actuator_ids"].astype(str).tolist())
            current_setting = raw["current_setting"].astype(np.float32)
        if state.shape[0] != times.size or current_setting.shape[0] != times.size:
            raise ValueError("compact trajectory time dimensions do not align")
        if np.any(np.diff(times) <= 0):
            raise ValueError("trajectory times must be strictly increasing")
        rows = pd.DataFrame(
            {
                "elapsed_seconds": times,
                "network_max_depth_m": state[..., 0].max(axis=1),
                "network_mean_depth_m": state[..., 0].mean(axis=1),
                "network_total_flood_rate_m3s": np.clip(
                    state[..., 2], 0.0, None
                ).sum(axis=1),
            }
        )
        return rows, actuator_ids, current_setting, meta

    node = pd.read_csv(root / str(meta["node_file"]), compression="infer")
    act = pd.read_csv(root / str(meta["actuator_file"]), compression="infer")
    flow_units = str(meta["flow_units"])
    system_units = str(meta["system_units"])
    rows: list[dict[str, float]] = []
    for elapsed, group in node.groupby("elapsed_seconds", sort=True):
        depth = length_to_m(group["depth"].to_numpy(dtype=float), system_units)
        flooding = flow_rate_to_m3s(
            group["flooding"].to_numpy(dtype=float), flow_units
        )
        rows.append(
            {
                "elapsed_seconds": int(elapsed),
                "network_max_depth_m": float(np.max(depth)),
                "network_mean_depth_m": float(np.mean(depth)),
                "network_total_flood_rate_m3s": float(
                    np.clip(flooding, 0.0, None).sum()
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("elapsed_seconds").reset_index(drop=True)
    actuator_ids = tuple(act["actuator_id"].astype(str).drop_duplicates().tolist())
    times = table["elapsed_seconds"].astype(int).to_numpy()
    setting = np.empty((len(times), len(actuator_ids)), dtype=np.float32)
    for i, elapsed in enumerate(times):
        at = act[act["elapsed_seconds"].astype(int) == int(elapsed)]
        values = at.set_index(at["actuator_id"].astype(str))[
            "current_setting"
        ].astype(float)
        setting[i] = [float(values.loc[a]) for a in actuator_ids]
    return table, actuator_ids, setting, meta


def _strata(score: np.ndarray) -> np.ndarray:
    if len(score) < 4 or np.allclose(score, score[0]):
        return np.zeros(len(score), dtype=int)
    q = np.quantile(score, [0.25, 0.50, 0.75])
    return np.digitize(score, q, right=True)


def design_checkpoints(
    run_index: pd.DataFrame,
    *,
    checkpoints_per_event: int = 8,
    minimum_elapsed_minutes: int = 60,
    seed: int = 42,
    allowed_splits: tuple[str, ...] = (
        "development",
        "calibration",
        "safety_audit",
    ),
) -> pd.DataFrame:
    required = {"metadata_path", "event_id", "rainfall_group", "scientific_split"}
    missing = sorted(required - set(run_index.columns))
    if missing:
        raise ValueError(f"No-control D0 run index missing columns: {missing}")
    if checkpoints_per_event < 4:
        raise ValueError("use at least four checkpoints per event to cover hydraulic strata")
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for _, item in run_index.iterrows():
        split = str(item["scientific_split"])
        if split == "final" or split not in allowed_splits:
            continue
        metadata_path = str(item["metadata_path"])
        state, actuator_ids, settings_by_time, meta = _state_table(metadata_path)
        _assert_replayable_no_control_prefix(meta, metadata_path)
        source_event_inp = _source_event_inp(item, meta)
        elapsed_all = state["elapsed_seconds"].to_numpy(dtype=int)
        keep = (
            (elapsed_all >= minimum_elapsed_minutes * 60)
            & (elapsed_all % 60 == 0)
        )
        state = state.loc[keep].copy()
        settings_by_time = settings_by_time[keep]
        if state.empty:
            raise ValueError(
                f"no exact minute-aligned history-ready checkpoint for event {item['event_id']}"
            )

        def scaled(values: np.ndarray) -> np.ndarray:
            lo, hi = float(np.min(values)), float(np.max(values))
            return (
                np.zeros_like(values)
                if hi <= lo
                else (values - lo) / (hi - lo)
            )

        score = 0.5 * scaled(state["network_max_depth_m"].to_numpy()) + 0.5 * scaled(
            state["network_total_flood_rate_m3s"].to_numpy()
        )
        state = state.reset_index(drop=True)
        state["hydraulic_stratum"] = _strata(score)
        chosen: list[int] = []
        per_stratum = max(1, checkpoints_per_event // 4)
        for stratum in range(4):
            candidates = state.index[
                state["hydraulic_stratum"] == stratum
            ].to_numpy()
            if candidates.size:
                chosen.extend(
                    rng.choice(
                        candidates,
                        size=min(per_stratum, candidates.size),
                        replace=False,
                    ).tolist()
                )
        remaining = checkpoints_per_event - len(set(chosen))
        if remaining > 0:
            pool = np.array(sorted(set(state.index) - set(chosen)), dtype=int)
            if pool.size:
                chosen.extend(
                    rng.choice(
                        pool, size=min(remaining, pool.size), replace=False
                    ).tolist()
                )

        for idx in sorted(set(chosen)):
            row = state.loc[idx]
            elapsed = int(row["elapsed_seconds"])
            if elapsed % 60:
                raise RuntimeError("checkpoint selection produced non-minute-aligned time")
            record: dict[str, object] = {
                "checkpoint_id": f"{item['event_id']}:t{elapsed}",
                "checkpoint_minutes": elapsed // 60,
                "checkpoint_elapsed_seconds": elapsed,
                "prefix_contract": "EXACT_NO_CONTROL_PREFIX_REPLAY_V1",
                "event_id": str(item["event_id"]),
                "rainfall_group": str(item["rainfall_group"]),
                "scientific_split": split,
                "development_fold": str(item.get("development_fold", "")),
                "inp_path": source_event_inp,
                "trajectory_metadata_path": metadata_path,
                "reference_swmm_engine_version": str(meta.get("swmm_engine_version", "")),
                "network_max_depth_m": float(row["network_max_depth_m"]),
                "network_total_flood_rate_m3s": float(
                    row["network_total_flood_rate_m3s"]
                ),
                "hydraulic_stratum": int(row["hydraulic_stratum"]),
            }
            for ai, aid in enumerate(actuator_ids):
                record[f"setting:{aid}"] = float(settings_by_time[idx, ai])
            output.append(record)
    result = pd.DataFrame(output)
    if result.empty:
        raise ValueError("checkpoint design produced no rows")
    if (result["scientific_split"].astype(str) == "final").any():
        raise RuntimeError("Final rainfall truth leaked into checkpoint design")
    if result["checkpoint_id"].duplicated().any():
        raise ValueError("duplicate checkpoint IDs")
    if (result["reference_swmm_engine_version"].astype(str).str.len() == 0).any():
        raise ValueError("checkpoint design lacks SWMM engine lineage")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Design stratified, exactly replay-verifiable No-control D2/D3 checkpoints"
    )
    parser.add_argument(
        "--run-index", required=True, help="prefer BASELINE_CACHE/NO_CONTROL_D0_INDEX.csv"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints-per-event", type=int, default=8)
    parser.add_argument("--minimum-elapsed-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = design_checkpoints(
        pd.read_csv(args.run_index),
        checkpoints_per_event=args.checkpoints_per_event,
        minimum_elapsed_minutes=args.minimum_elapsed_minutes,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(
        json.dumps(
            {
                "checkpoints": len(frame),
                "events": int(frame["event_id"].nunique()),
                "rainfall_groups": int(frame["rainfall_group"].nunique()),
                "prefix_contract": "EXACT_NO_CONTROL_PREFIX_REPLAY_V1",
                "splits": frame.groupby("scientific_split")["checkpoint_id"].count().to_dict(),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
