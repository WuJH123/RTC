from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFractions:
    """Legacy generic split helper retained only for non-Formal utilities/tests.

    **Do not use this helper for Project7 v0.6.9 Formal execution.** The public split CLI has
    been removed and the active 30-event allocation is fixed by
    ``configs/project7_v069_split_contract.json``. These historical four-role fractions remain
    solely so generic workspace/regression utilities that are intentionally cohort-size agnostic
    can still construct synthetic data.
    """

    development: float = 0.60
    calibration: float = 0.15
    safety_audit: float = 0.10
    final: float = 0.15

    def validate(self) -> None:
        values = (self.development, self.calibration, self.safety_audit, self.final)
        if any(v <= 0 for v in values):
            raise ValueError("all legacy generic split fractions must be positive")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("legacy generic split fractions must sum to 1")


def assign_rainfall_group_splits(
    frame: pd.DataFrame,
    *,
    rainfall_group_col: str = "rainfall_group",
    seed: int = 42,
    fractions: SplitFractions = SplitFractions(),
    development_validation_fraction: float = 0.20,
) -> pd.DataFrame:
    """Legacy synthetic/generic splitter; forbidden for the active v0.6.9 Formal registry."""

    fractions.validate()
    if not 0.0 < development_validation_fraction < 1.0:
        raise ValueError("development_validation_fraction must lie in (0,1)")
    if rainfall_group_col not in frame.columns:
        raise ValueError(f"missing rainfall group column: {rainfall_group_col}")
    groups = pd.Series(frame[rainfall_group_col].dropna().astype(str).unique())
    if len(groups) < 5:
        raise ValueError("at least five rainfall groups are required")
    rng = np.random.default_rng(seed)
    shuffled = groups.to_numpy(copy=True)
    rng.shuffle(shuffled)

    n = len(shuffled)
    raw = np.array(
        [fractions.development, fractions.calibration, fractions.safety_audit, fractions.final]
    ) * n
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        reducible = np.where(counts > 1, counts, -1)
        idx = int(np.argmax(reducible))
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

    development_groups = [g for g in shuffled if mapping[str(g)] == "development"]
    if len(development_groups) < 2:
        raise ValueError("development split needs at least two rainfall groups for train/validation")
    dev_rng = np.random.default_rng(seed + 1)
    dev_shuffled = np.asarray(development_groups, dtype=object)
    dev_rng.shuffle(dev_shuffled)
    n_validation = max(1, int(round(len(dev_shuffled) * development_validation_fraction)))
    n_validation = min(n_validation, len(dev_shuffled) - 1)
    validation_groups = {str(g) for g in dev_shuffled[:n_validation]}

    out = frame.copy()
    group_text = out[rainfall_group_col].astype(str)
    out["scientific_split"] = group_text.map(mapping)
    if out["scientific_split"].isna().any():
        raise ValueError("null/unknown rainfall groups cannot be assigned")
    out["development_fold"] = ""
    is_dev = out["scientific_split"] == "development"
    out.loc[is_dev, "development_fold"] = np.where(
        group_text[is_dev].isin(validation_groups), "validation", "train"
    )
    verify_disjoint_rainfall_splits(out, rainfall_group_col=rainfall_group_col)
    verify_development_folds(out, rainfall_group_col=rainfall_group_col)
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
        raise ValueError(f"missing legacy generic scientific split roles: {missing}")


def verify_development_folds(
    frame: pd.DataFrame,
    *,
    rainfall_group_col: str = "rainfall_group",
) -> None:
    if "development_fold" not in frame.columns:
        raise ValueError("development_fold is required")
    dev = frame[frame["scientific_split"] == "development"]
    present = set(dev["development_fold"].astype(str))
    if present != {"train", "validation"}:
        raise ValueError(f"development folds must be train/validation, got {sorted(present)}")
    cross = dev.groupby(rainfall_group_col)["development_fold"].nunique()
    bad = cross[cross != 1]
    if not bad.empty:
        raise ValueError(f"rainfall-group leakage across development folds: {bad.index.tolist()[:10]}")
    non_dev = frame[frame["scientific_split"] != "development"]
    if (non_dev["development_fold"].astype(str) != "").any():
        raise ValueError("non-development rows must not carry a development fold")
