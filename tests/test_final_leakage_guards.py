from __future__ import annotations

import pandas as pd
import pytest

from rtc.large_data_cli import _reject_final_rows


def test_lower_level_prelock_generators_reject_final_rows() -> None:
    frame = pd.DataFrame(
        [
            {"event_id": "dev", "scientific_split": "development"},
            {"event_id": "final", "scientific_split": "final"},
        ]
    )
    with pytest.raises(ValueError, match="refuses 1 Final rows"):
        _reject_final_rows(frame, context="pilot")


def test_lower_level_generator_requires_split_lineage() -> None:
    with pytest.raises(ValueError, match="requires scientific_split lineage"):
        _reject_final_rows(pd.DataFrame([{"event_id": "e1"}]), context="pilot")
