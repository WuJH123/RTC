from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFractions:
    development: float = 0.60
    calibration: float = 0.15
    safety_audit: float = 0.10
    final: float = 0.15

    def validate(self) -> None:
        values = (self.development, self.calibration, self.safety_audit, self.final)
        if any(v <= 0 for v in values):
            raise ValueError("all split fractions must be positive")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("split fractions must sum to 1")


def assign_rainfall_group_splits(
    frame: pd.DataFrame,
    *,
    rainfall_group_col: str = "rainfall_group",
    seed: int = 42,
    fractions: SplitFractions = SplitFractions(),
) -> pd.DataFrame:
    """Assign entire rainfall groups to mutually exclusive scientific roles.

    Rows from the same rainfall group can never cross development, calibration,
    independent safety-audit, or untouched final partitions.
    """

    fractions.validate()
    if rainfall_group_col not in frame.columns:
        raise ValueError(f"missing rainfall group column: {rainfall_group_col}")
    groups = pd.Series(frame[rainfall_group_col].dropna().astype(str).unique())
    if len(groups) < 4:
        raise ValueError("at least four rainfall groups are required")
    rng = np.random.default_rng(seed)
    shuffled = groups.to_numpy(copy=True)
    rng.shuffle(shuffled)

    n = len(shuffled)
    # Largest-remainder allocation with at least one group per role.
    raw = np.array(
        [fractions.development, fractions.calibration, fractions.safety_audit, fractions.final]
    ) * n
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        idx = int(np.argmax(counts - 1))
        counts[idx] -= 1
    while counts.sum() < n:
        remainder = raw - counts
        counts[int(np.argmax(remainder))] += 1

    roles = ("development", "calibration", "safety_audit", "final")
    mapping: dict[str, str] = {}
    start = 0
    for role, count in zip(roles, counts, strict=True):
        for group in shuffled[start : start + count]:
            mapping[str(group)] = role
        start += count

    out = frame.copy()
    out["scientific_split"] = out[rainfall_group_col].astype(str).map(mapping)
    if out["scientific_split"].isna().any():
        raise ValueError("null/unknown rainfall groups cannot be assigned")
    verify_disjoint_rainfall_splits(out, rainfall_group_col=rainfall_group_col)
    return out


def verify_disjoint_rainfall_splits(
    frame: pd.DataFrame,
    *,
    rainfall_group_col: str = "rainfall_group",
    split_col: str = "scientific_split",
) -> None:
    if rainfall_group_col not in frame.columns or split_col not in frame.columns:
        raise ValueError("rainfall_group and scientific_split columns are required")
    cross = frame.groupby(rainfall_group_col, dropna=False)[split_col].nunique(dropna=False)
    bad = cross[cross != 1]
    if not bad.empty:
        raise ValueError(f"rainfall-group leakage across splits: {bad.index.tolist()[:10]}")
    required = {"development", "calibration", "safety_audit", "final"}
    present = set(frame[split_col].dropna().astype(str))
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"missing scientific split roles: {missing}")
