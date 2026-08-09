from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORCING_DESCRIPTOR_COLUMNS = (
    "total_depth_mm",
    "duration_minutes",
    "peak_intensity_mmhr",
    "antecedent_rainfall_mm",
)


def _usable_descriptor_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    usable: list[str] = []
    for column in FORCING_DESCRIPTOR_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all():
            usable.append(column)
    return tuple(usable)


def _diverse_group_indices(values: np.ndarray, count: int, *, seed: int) -> list[int]:
    """Greedy farthest-point sampling in standardized forcing-descriptor space."""

    if values.ndim != 2 or values.shape[0] < count or count <= 0:
        raise ValueError("invalid Phase-0 descriptor matrix/count")
    if values.shape[1] == 0:
        rng = np.random.default_rng(seed)
        return rng.choice(values.shape[0], size=count, replace=False).astype(int).tolist()

    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (values - center) / scale

    distance_to_center = np.linalg.norm(z, axis=1)
    first = int(np.argmax(distance_to_center))
    selected = [first]
    remaining = set(range(values.shape[0])) - {first}
    while len(selected) < count:
        best_idx = None
        best_distance = -np.inf
        for idx in sorted(remaining):
            min_distance = min(float(np.linalg.norm(z[idx] - z[j])) for j in selected)
            if min_distance > best_distance + 1e-12:
                best_distance = min_distance
                best_idx = idx
        assert best_idx is not None
        selected.append(int(best_idx))
        remaining.remove(best_idx)
    return selected


def design_phase0_events(
    events: pd.DataFrame,
    *,
    rainfall_groups: int = 8,
    development_fold: str = "train",
    seed: int = 42,
    one_event_per_group: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a small development-only Phase-0 cohort using forcing information only.

    Hydraulic outcomes are deliberately unavailable to this selector. When rainfall forcing
    descriptors exist, groups are chosen by deterministic farthest-point coverage in forcing
    space. If no supported descriptors exist, the fallback is a seeded group-only sample.
    """

    required = {"event_id", "rainfall_group", "inp_path", "scientific_split", "development_fold"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Phase-0 event registry missing columns: {missing}")
    if rainfall_groups <= 0:
        raise ValueError("rainfall_groups must be positive")
    if development_fold not in {"train", "validation", "all"}:
        raise ValueError("development_fold must be train, validation or all")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("event_id must be unique")

    eligible = events[events["scientific_split"].astype(str) == "development"].copy()
    if development_fold != "all":
        eligible = eligible[eligible["development_fold"].astype(str) == development_fold].copy()
    if eligible.empty:
        raise ValueError("no eligible development rows for Phase-0")

    descriptors = _usable_descriptor_columns(eligible)
    grouped_records: list[dict[str, object]] = []
    for rainfall_group, group in eligible.groupby("rainfall_group", sort=True):
        record: dict[str, object] = {"rainfall_group": str(rainfall_group)}
        for column in descriptors:
            record[column] = float(pd.to_numeric(group[column], errors="raise").median())
        grouped_records.append(record)
    group_frame = pd.DataFrame(grouped_records).sort_values("rainfall_group").reset_index(drop=True)
    if len(group_frame) < rainfall_groups:
        raise ValueError(
            f"requested {rainfall_groups} Phase-0 rainfall groups but only {len(group_frame)} are eligible"
        )

    matrix = (
        group_frame[list(descriptors)].to_numpy(dtype=float)
        if descriptors
        else np.empty((len(group_frame), 0), dtype=float)
    )
    selected_indices = _diverse_group_indices(matrix, rainfall_groups, seed=seed)
    selected_groups = group_frame.iloc[selected_indices]["rainfall_group"].astype(str).tolist()

    selected = eligible[eligible["rainfall_group"].astype(str).isin(selected_groups)].copy()
    if one_event_per_group:
        selected = (
            selected.sort_values(["rainfall_group", "event_id"])
            .groupby("rainfall_group", as_index=False, sort=True)
            .head(1)
        )
    selected = selected.sort_values(["rainfall_group", "event_id"]).reset_index(drop=True)
    if selected["rainfall_group"].astype(str).nunique() != rainfall_groups:
        raise RuntimeError("Phase-0 selection did not preserve the requested rainfall-group count")
    if (selected["scientific_split"].astype(str) != "development").any():
        raise RuntimeError("non-development row leaked into Phase-0")

    summary: dict[str, object] = {
        "contract": "PHASE0_FORCING_ONLY_GROUP_SELECTION_V1",
        "rainfall_groups_requested": int(rainfall_groups),
        "rainfall_groups_selected": int(selected["rainfall_group"].astype(str).nunique()),
        "event_rows_selected": int(len(selected)),
        "development_fold": development_fold,
        "one_event_per_group": bool(one_event_per_group),
        "seed": int(seed),
        "forcing_descriptor_columns": list(descriptors),
        "selection_mode": (
            "DETERMINISTIC_FARTHEST_POINT_FORCING_COVERAGE"
            if descriptors
            else "SEEDED_GROUP_ONLY_FALLBACK_NO_FORCING_DESCRIPTORS"
        ),
        "hydraulic_outcomes_used": False,
        "selected_rainfall_groups": selected_groups,
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a small development-only, forcing-diverse Phase-0 rainfall cohort"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--development-fold", choices=["train", "validation", "all"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-events-per-group", action="store_true")
    args = parser.parse_args()

    selected, summary = design_phase0_events(
        pd.read_csv(args.events),
        rainfall_groups=args.groups,
        development_fold=args.development_fold,
        seed=args.seed,
        one_event_per_group=not args.all_events_per_group,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary["out"] = str(out)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
