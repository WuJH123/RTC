from __future__ import annotations

import pandas as pd

from rtc.formal_final_v4 import _group_detail


def test_event_variants_do_not_overweight_one_rainfall_group() -> None:
    detail = pd.DataFrame(
        [
            {"rainfall_group": "g1", "strategy": "proposed", "tfv_m3": 0.0},
            {"rainfall_group": "g1", "strategy": "proposed", "tfv_m3": 100.0},
            {"rainfall_group": "g1", "strategy": "proposed", "tfv_m3": 200.0},
            {"rainfall_group": "g2", "strategy": "proposed", "tfv_m3": 1000.0},
        ]
    )
    grouped = _group_detail(detail, ["tfv_m3"])
    assert len(grouped) == 2
    assert float(grouped.loc[grouped["rainfall_group"] == "g1", "tfv_m3"].iloc[0]) == 100.0
    # Formal mean is the mean of independent group means: (100 + 1000)/2 = 550,
    # not the raw-event mean (0+100+200+1000)/4 = 325.
    assert float(grouped["tfv_m3"].mean()) == 550.0
