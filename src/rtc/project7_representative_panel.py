"""Outcome-independent representative-event selection for Project7 evaluation.

The selector is intentionally based only on exogenous event descriptors.  PFV, TFV, peak, action
counts, and strategy wins/losses are ignored, so an already-viewed evaluation can be used to diagnose
policy design without allowing favorable outcomes to determine which events enter the paper panel.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd


REPRESENTATIVE_PANEL_CONTRACT = "PROJECT7_OUTCOME_INDEPENDENT_REPRESENTATIVE_PANEL_V1"
DEFAULT_REPRESENTATIVE_EVENT_COUNT = 21

_RP = re.compile(r"(?:^|_)RP0*(\d+)(?:_|$)", re.IGNORECASE)
_DURATION = re.compile(r"(?:^|_)D0*(\d+)(?:_|$)", re.IGNORECASE)
_RATIO = re.compile(r"(?:^|_)R(?!P)0*(\d+)(?:_|$)", re.IGNORECASE)
_FAMILY = re.compile(r"^(.*?)(?=_RP\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RepresentativePanel:
    contract: str
    selected_event_ids: tuple[str, ...]
    input_event_count: int
    target_event_count: int
    family_counts: dict[str, int]
    descriptor_columns: tuple[str, ...]


def _number_from_id(pattern: re.Pattern[str], event_id: str) -> float:
    match = pattern.search(event_id)
    return float(match.group(1)) if match else float("nan")


def _family_from_id(event_id: str) -> str:
    match = _FAMILY.search(event_id)
    if match and match.group(1):
        return match.group(1).strip("_").upper()
    tokens = event_id.split("_")
    return (tokens[0] if tokens else event_id).upper()


def _first_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.iloc[0]) if len(values) else float("nan")


def _event_table(rows: pd.DataFrame, *, event_id_column: str) -> pd.DataFrame:
    if event_id_column not in rows.columns:
        raise ValueError(f"missing event ID column: {event_id_column}")
    table = rows.copy()
    table[event_id_column] = table[event_id_column].astype(str).str.strip()
    table = table[table[event_id_column] != ""]
    if table.empty:
        raise ValueError("representative-panel input contains no event IDs")

    # Completion is provenance, not a hydraulic outcome.  If cell-level rows are supplied, an event
    # is eligible only when every supplied strategy cell is marked complete.
    if "completion_pass" in table.columns:
        completion = table["completion_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
        table = table.assign(_completion_pass=completion)
        good = table.groupby(event_id_column, sort=False)["_completion_pass"].all()
        eligible_ids = set(good[good].index.astype(str))
        table = table[table[event_id_column].isin(eligible_ids)]
    if table.empty:
        raise ValueError("no fully completed events are available for representative selection")

    records: list[dict[str, object]] = []
    for event_id, group in table.groupby(event_id_column, sort=True):
        event_id = str(event_id)
        rp = (
            _first_finite(group["return_period_year"])
            if "return_period_year" in group.columns
            else _number_from_id(_RP, event_id)
        )
        duration = (
            _first_finite(group["duration_minutes"])
            if "duration_minutes" in group.columns
            else _number_from_id(_DURATION, event_id)
        )
        ratio = (
            _first_finite(group["rainfall_ratio"])
            if "rainfall_ratio" in group.columns
            else _number_from_id(_RATIO, event_id)
        )
        records.append(
            {
                "event_id": event_id,
                "family": _family_from_id(event_id),
                "return_period_year": rp,
                "duration_minutes": duration,
                "rainfall_ratio": ratio,
            }
        )
    result = pd.DataFrame.from_records(records).sort_values("event_id").reset_index(drop=True)
    if not np.isfinite(result[["return_period_year", "duration_minutes"]].to_numpy()).all():
        raise ValueError(
            "every event must expose return period and duration either as columns or in its event ID"
        )
    return result


def _scaled_numeric(table: pd.DataFrame) -> np.ndarray:
    columns = ["return_period_year", "duration_minutes"]
    if np.isfinite(table["rainfall_ratio"].to_numpy()).any():
        columns.append("rainfall_ratio")
    matrix = table[columns].to_numpy(dtype=np.float64, copy=True)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if not finite.all():
            fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
            values[~finite] = fill
        lo, hi = float(np.min(values)), float(np.max(values))
        matrix[:, column] = 0.0 if hi <= lo else (values - lo) / (hi - lo)
    return matrix


def _feature_matrix(table: pd.DataFrame) -> np.ndarray:
    numeric = _scaled_numeric(table)
    families = tuple(sorted(table["family"].astype(str).unique()))
    one_hot = np.zeros((len(table), len(families)), dtype=np.float64)
    family_index = {name: i for i, name in enumerate(families)}
    for row, family in enumerate(table["family"].astype(str)):
        one_hot[row, family_index[family]] = 1.0
    # Family membership is important for coverage but should not overwhelm rainfall geometry.
    return np.concatenate([numeric, 0.75 * one_hot], axis=1)


def _nearest_to_family_center(table: pd.DataFrame, features: np.ndarray, family: str) -> int:
    indices = np.flatnonzero(table["family"].to_numpy() == family)
    center = np.median(features[indices], axis=0)
    distance = np.linalg.norm(features[indices] - center, axis=1)
    order = sorted(zip(distance.tolist(), indices.tolist()), key=lambda item: (item[0], str(table.iloc[item[1]]["event_id"])))
    return int(order[0][1])


def select_representative_panel(
    rows: pd.DataFrame,
    *,
    target_event_count: int = DEFAULT_REPRESENTATIVE_EVENT_COUNT,
    event_id_column: str = "event_id",
    training_event_ids: Iterable[str] = (),
) -> RepresentativePanel:
    """Select a deterministic space-filling panel using exogenous descriptors only."""
    table = _event_table(rows, event_id_column=event_id_column)
    n = len(table)
    target = int(target_event_count)
    if target <= 0:
        raise ValueError("target_event_count must be positive")
    target = min(target, n)

    training = {str(value).strip().lower() for value in training_event_ids if str(value).strip()}
    overlap = [event_id for event_id in table["event_id"] if event_id.lower() in training]
    if overlap:
        raise ValueError(
            "training/evaluation event leakage detected: " + ",".join(sorted(overlap)[:8])
        )

    features = _feature_matrix(table)
    selected: list[int] = []

    # First guarantee broad event-family representation.  If target is smaller than the number of
    # families, favor larger families and use a deterministic tie-break.
    family_sizes = table["family"].value_counts().to_dict()
    families = sorted(family_sizes, key=lambda value: (-int(family_sizes[value]), str(value)))
    for family in families[:target]:
        selected.append(_nearest_to_family_center(table, features, family))

    # Preserve global rainfall extremes because these are commonly the most difficult hydraulic
    # regimes and would otherwise be easy for a center-based sample to miss.
    for column in ("return_period_year", "duration_minutes", "rainfall_ratio"):
        values = table[column].to_numpy(dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            continue
        for index in (finite[int(np.argmin(values[finite]))], finite[int(np.argmax(values[finite]))]):
            if index not in selected and len(selected) < target:
                selected.append(int(index))

    # Greedy farthest-point completion approximates a k-medoids/space-filling panel without adding
    # a clustering dependency.  Ties are resolved by event ID for exact reproducibility.
    all_indices = set(range(n))
    while len(selected) < target:
        remaining = sorted(all_indices.difference(selected))
        best_index = None
        best_distance = -1.0
        for index in remaining:
            distance = min(
                float(np.linalg.norm(features[index] - features[chosen])) for chosen in selected
            ) if selected else float("inf")
            event_id = str(table.iloc[index]["event_id"])
            if (
                distance > best_distance + 1.0e-12
                or (
                    abs(distance - best_distance) <= 1.0e-12
                    and (best_index is None or event_id < str(table.iloc[best_index]["event_id"]))
                )
            ):
                best_index = int(index)
                best_distance = float(distance)
        assert best_index is not None
        selected.append(best_index)

    selected_ids = tuple(str(table.iloc[index]["event_id"]) for index in selected)
    selected_table = table[table["event_id"].isin(selected_ids)]
    family_counts = {
        str(key): int(value)
        for key, value in selected_table["family"].value_counts().sort_index().items()
    }
    return RepresentativePanel(
        contract=REPRESENTATIVE_PANEL_CONTRACT,
        selected_event_ids=selected_ids,
        input_event_count=n,
        target_event_count=target,
        family_counts=family_counts,
        descriptor_columns=("event_id", "family", "return_period_year", "duration_minutes", "rainfall_ratio"),
    )


def selected_rows(rows: pd.DataFrame, panel: RepresentativePanel, *, event_id_column: str = "event_id") -> pd.DataFrame:
    selected = set(panel.selected_event_ids)
    result = rows[rows[event_id_column].astype(str).isin(selected)].copy()
    order = {event_id: i for i, event_id in enumerate(panel.selected_event_ids)}
    result["representative_panel_order"] = result[event_id_column].astype(str).map(order)
    return result.sort_values(["representative_panel_order", event_id_column]).reset_index(drop=True)


__all__ = [
    "DEFAULT_REPRESENTATIVE_EVENT_COUNT",
    "REPRESENTATIVE_PANEL_CONTRACT",
    "RepresentativePanel",
    "select_representative_panel",
    "selected_rows",
]
