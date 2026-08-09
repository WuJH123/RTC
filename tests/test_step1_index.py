from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rtc.step1_index import build_step1_index


def test_step1_index_keeps_group_disjoint_validation_and_train_only_d1(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e_train",
                "rainfall_group": "g_train",
                "scientific_split": "development",
                "development_fold": "train",
                "strategy": "no_control",
                "metadata_path": "base_train.json",
            },
            {
                "event_id": "e_val",
                "rainfall_group": "g_val",
                "scientific_split": "development",
                "development_fold": "validation",
                "strategy": "internal_rtc",
                "metadata_path": "base_val.json",
            },
        ]
    ).to_csv(baseline, index=False)
    d1 = tmp_path / "d1.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e_d1",
                "rainfall_group": "g_d1",
                "scientific_split": "development",
                "development_fold": "train",
                "strategy": "d1_exploration",
                "metadata_path": "d1.json",
            }
        ]
    ).to_csv(d1, index=False)
    combined, train, validation = build_step1_index(
        baseline_index_path=baseline, d1_index_path=d1
    )
    assert len(combined) == 3
    assert set(train["rainfall_group"]) == {"g_train", "g_d1"}
    assert set(validation["rainfall_group"]) == {"g_val"}
    assert set(validation["source_role"]) == {"BASELINE"}


def test_step1_index_rejects_d1_validation_leakage(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e_train",
                "rainfall_group": "g_train",
                "scientific_split": "development",
                "development_fold": "train",
                "strategy": "no_control",
                "metadata_path": "base_train.json",
            },
            {
                "event_id": "e_val",
                "rainfall_group": "g_val",
                "scientific_split": "development",
                "development_fold": "validation",
                "strategy": "no_control",
                "metadata_path": "base_val.json",
            },
        ]
    ).to_csv(baseline, index=False)
    d1 = tmp_path / "d1.csv"
    pd.DataFrame(
        [
            {
                "event_id": "e_bad",
                "rainfall_group": "g_other",
                "scientific_split": "development",
                "development_fold": "validation",
                "metadata_path": "bad.json",
            }
        ]
    ).to_csv(d1, index=False)
    with pytest.raises(ValueError, match="development/train-only"):
        build_step1_index(baseline_index_path=baseline, d1_index_path=d1)
