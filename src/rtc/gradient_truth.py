from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GradientAgreement:
    samples: int
    nonzero_truth: int
    sign_accuracy: float
    cosine_similarity: float
    magnitude_mae: float


def central_difference(lower_value: np.ndarray, upper_value: np.ndarray, lower_setting: np.ndarray, upper_setting: np.ndarray) -> np.ndarray:
    lo_v = np.asarray(lower_value, dtype=float)
    hi_v = np.asarray(upper_value, dtype=float)
    lo_u = np.asarray(lower_setting, dtype=float)
    hi_u = np.asarray(upper_setting, dtype=float)
    denom = hi_u - lo_u
    if np.any(np.abs(denom) <= 1e-12):
        raise ValueError("finite-difference settings must differ")
    return (hi_v - lo_v) / denom


def compare_gradient_vectors(predicted: np.ndarray, swmm_truth: np.ndarray, *, zero_tolerance: float = 1e-9) -> GradientAgreement:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    truth = np.asarray(swmm_truth, dtype=float).reshape(-1)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("gradient vectors must have the same non-zero length")
    valid = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[valid], truth[valid]
    nz = np.abs(truth) > zero_tolerance
    if not np.any(nz):
        sign_accuracy = float("nan")
    else:
        sign_accuracy = float(np.mean(np.sign(pred[nz]) == np.sign(truth[nz])))
    denom = float(np.linalg.norm(pred) * np.linalg.norm(truth))
    cosine = float(np.dot(pred, truth) / denom) if denom > 0 else float("nan")
    return GradientAgreement(
        samples=int(pred.size),
        nonzero_truth=int(nz.sum()),
        sign_accuracy=sign_accuracy,
        cosine_similarity=cosine,
        magnitude_mae=float(np.mean(np.abs(pred - truth))),
    )


def build_d2_finite_difference_table(
    manifest: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    metric_col: str,
) -> pd.DataFrame:
    """Compile same-checkpoint central/one-sided finite differences from D2 branches.

    ``outcome`` must contain ``candidate_action_sha256`` and the requested metric derived
    from authoritative SWMM. The function groups by checkpoint and actuator, never across
    rainfall states.
    """

    required_manifest = {
        "checkpoint_id",
        "actuator_id",
        "base_setting",
        "requested_setting",
        "candidate_action_sha256",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if "candidate_action_sha256" not in outcome.columns or metric_col not in outcome.columns:
        raise ValueError("outcome missing candidate_action_sha256 or requested metric")
    merged = manifest.merge(outcome[["candidate_action_sha256", metric_col]], on="candidate_action_sha256", how="inner")
    rows: list[dict[str, object]] = []
    group_cols = ["checkpoint_id", "actuator_id"]
    for keys, group in merged.groupby(group_cols, sort=False):
        group = group.sort_values("requested_setting")
        base = float(group["base_setting"].iloc[0])
        below = group[group["requested_setting"] < base]
        above = group[group["requested_setting"] > base]
        if not below.empty and not above.empty:
            lo, hi = below.iloc[-1], above.iloc[0]
        else:
            center = group[np.isclose(group["requested_setting"], base)]
            side = above.iloc[[0]] if not above.empty else below.iloc[[-1]]
            if center.empty or side.empty:
                continue
            lo, hi = (center.iloc[0], side.iloc[0]) if float(side.iloc[0]["requested_setting"]) > base else (side.iloc[0], center.iloc[0])
        du = float(hi["requested_setting"] - lo["requested_setting"])
        if abs(du) <= 1e-12:
            continue
        rows.append(
            {
                "checkpoint_id": keys[0],
                "actuator_id": keys[1],
                "lower_setting": float(lo["requested_setting"]),
                "upper_setting": float(hi["requested_setting"]),
                "swmm_gradient": float((hi[metric_col] - lo[metric_col]) / du),
                "metric": metric_col,
            }
        )
    return pd.DataFrame.from_records(rows)
