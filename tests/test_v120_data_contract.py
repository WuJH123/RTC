from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rtc.step2_v120_data_contract import (
    D2_GROUPS,
    D3_GROUPS,
    INTERNAL_HOLDOUT_FRACTION,
    validate_canonical_cache_population,
    validate_internal_holdout_fraction,
)


def _cache(d2_count: int = D2_GROUPS, d3_count: int = D3_GROUPS):
    truth = np.ones((25, 2), dtype=np.float32)
    entries = {}
    for source, count in (("D2", d2_count), ("D3", d3_count)):
        for i in range(count):
            entries[f"{source}::{i}"] = SimpleNamespace(
                indices=tuple(range(25)),
                reference_index=0,
                arrays={"exact_node_flood_volume_m3": truth},
            )
    return SimpleNamespace(entry=lambda name: entries[name])


def test_canonical_population_is_3600_per_source() -> None:
    cache = _cache()
    result = validate_canonical_cache_population(
        cache,
        [f"D2::{i}" for i in range(D2_GROUPS)],
        [f"D3::{i}" for i in range(D3_GROUPS)],
    )
    assert result["d2_branches"] == 3600
    assert result["d3_branches"] == 3600
    assert result["d2_candidates"] == 3456
    assert result["d3_candidates"] == 3456


def test_canonical_population_rejects_missing_group() -> None:
    cache = _cache(d2_count=D2_GROUPS - 1)
    with pytest.raises(ValueError, match="144/144"):
        validate_canonical_cache_population(
            cache,
            [f"D2::{i}" for i in range(D2_GROUPS - 1)],
            [f"D3::{i}" for i in range(D3_GROUPS)],
        )


def test_internal_holdout_fraction_is_frozen() -> None:
    assert validate_internal_holdout_fraction(0.20) == INTERNAL_HOLDOUT_FRACTION
    with pytest.raises(ValueError, match="frozen at 0.20"):
        validate_internal_holdout_fraction(0.25)
