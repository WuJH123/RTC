"""Deterministic event-level statistics for Project7 publication evidence.

The rainfall event is the statistical unit. With Final6, the complete nonparametric bootstrap space
contains only 6**6 = 46,656 resamples, so percentile confidence intervals can be enumerated exactly
without Monte Carlo randomness. Exact two-sided sign tests are also reported for directional
consistency. These summaries are descriptive/inferential reporting aids, never controller gates.
"""
from __future__ import annotations

from itertools import product
from math import comb
from typing import Sequence


FINAL_EVENT_COUNT = 6


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")
    if probability <= 0.0:
        return float(sorted_values[0])
    if probability >= 1.0:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def exact_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    lower_probability: float = 0.025,
    upper_probability: float = 0.975,
) -> tuple[float, float]:
    data = tuple(float(value) for value in values)
    if len(data) != FINAL_EVENT_COUNT:
        raise ValueError("Project7 Final publication bootstrap requires exactly six events")
    means = []
    for indices in product(range(FINAL_EVENT_COUNT), repeat=FINAL_EVENT_COUNT):
        means.append(sum(data[index] for index in indices) / FINAL_EVENT_COUNT)
    means.sort()
    return (
        _percentile(means, lower_probability),
        _percentile(means, upper_probability),
    )


def exact_two_sided_sign_test_pvalue(*, wins: int, losses: int) -> float | None:
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("wins/losses must be non-negative")
    non_ties = wins + losses
    if non_ties == 0:
        return None
    tail_count = min(wins, losses)
    lower_tail = sum(comb(non_ties, k) for k in range(tail_count + 1)) / (2**non_ties)
    return float(min(1.0, 2.0 * lower_tail))


__all__ = [
    "FINAL_EVENT_COUNT",
    "exact_bootstrap_mean_ci",
    "exact_two_sided_sign_test_pvalue",
]
