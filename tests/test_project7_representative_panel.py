from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtc.project7_representative_panel import select_representative_panel


def _rows(n: int = 107) -> pd.DataFrame:
    families = ("FAD", "QMR_V14", "V11_FM", "V12")
    records = []
    for i in range(n):
        family = families[i % len(families)]
        rp = 2 + (i * 7) % 99
        duration = (90, 135, 165, 195, 225, 255, 285)[i % 7]
        records.append(
            {
                "event_id": f"{family}_RP{rp:03d}_D{duration:03d}_case{i:03d}",
                "return_period_year": rp,
                "duration_minutes": duration,
                "completion_pass": True,
                # Deliberately tempting outcome fields: the selector must never use them.
                "proposed_tfv_m3": float(100000 + 13 * i),
                "proposed_pfv_m3": float(1000 + 5 * i),
                "proposed_better_than_auto_rbc": bool(i % 3),
            }
        )
    return pd.DataFrame.from_records(records)


def test_default_style_107_to_21_is_deterministic_and_covers_families() -> None:
    rows = _rows()
    first = select_representative_panel(rows, target_event_count=21)
    second = select_representative_panel(rows.sample(frac=1.0, random_state=9), target_event_count=21)
    assert len(first.selected_event_ids) == 21
    assert first.selected_event_ids == second.selected_event_ids
    assert set(first.family_counts) == {"FAD", "QMR_V14", "V11_FM", "V12"}


def test_outcome_columns_cannot_change_selection() -> None:
    rows = _rows()
    baseline = select_representative_panel(rows, target_event_count=21).selected_event_ids
    changed = rows.copy()
    rng = np.random.default_rng(123)
    changed["proposed_tfv_m3"] = rng.normal(size=len(changed)) * 1.0e9
    changed["proposed_pfv_m3"] = rng.normal(size=len(changed)) * 1.0e8
    changed["proposed_better_than_auto_rbc"] = rng.integers(0, 2, size=len(changed)).astype(bool)
    assert select_representative_panel(changed, target_event_count=21).selected_event_ids == baseline


def test_training_evaluation_overlap_is_rejected() -> None:
    rows = _rows(12)
    with pytest.raises(ValueError, match="leakage"):
        select_representative_panel(
            rows,
            target_event_count=6,
            training_event_ids=[rows.iloc[4]["event_id"]],
        )


def test_incomplete_cell_excludes_whole_event() -> None:
    rows = pd.DataFrame(
        [
            {"event_id": "V12_RP010_D090_a", "completion_pass": True},
            {"event_id": "V12_RP010_D090_a", "completion_pass": False},
            {"event_id": "V12_RP020_D135_b", "completion_pass": True},
        ]
    )
    panel = select_representative_panel(rows, target_event_count=2)
    assert panel.selected_event_ids == ("V12_RP020_D135_b",)
