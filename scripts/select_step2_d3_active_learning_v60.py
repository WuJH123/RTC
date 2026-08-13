"""Select a bounded second-round TrainFit-only V6 D3 active-learning batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.step2_d3_design_v60 import select_active_learning_pool_v60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.candidate_scores)
    if "scientific_split" in frame.columns and {
        str(value).strip().lower() for value in frame["scientific_split"]
    } != {"development"}:
        raise ValueError("V6 active learning is development-only")
    if "development_fold" in frame.columns and {
        str(value).strip().lower() for value in frame["development_fold"]
    } != {"train"}:
        raise ValueError("V6 active learning is Train-only")
    if "v60_internal_role" not in frame.columns:
        raise ValueError(
            "V6 active learning requires v60_internal_role and may only select TrainFit; "
            "the frozen TrainInternalHoldout must remain untouched"
        )
    if {str(value).strip().lower() for value in frame["v60_internal_role"]} != {"fit"}:
        raise ValueError("V6 active learning refuses TrainInternalHoldout candidates")
    forbidden = {
        "authoritative_delta_tfv_m3",
        "true_delta_tfv_m3",
        "swmm_delta_tfv_m3",
        "candidate_tfv_m3",
    }
    if forbidden & set(frame.columns):
        raise ValueError(
            "active-learning score table contains authoritative future outcome columns; "
            "selection must be model/forcing/manifold-only"
        )
    selected = select_active_learning_pool_v60(frame, budget=args.budget)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)
    print(
        json.dumps(
            {
                "contract": "PROJECT7_STEP2_V60_ACTIVE_LEARNING_SELECTION_V1",
                "rows": int(len(selected)),
                "rainfall_groups": int(selected["rainfall_group"].nunique()),
                "internal_role": "fit",
                "internal_holdout_used": False,
                "out": str(out),
                "authoritative_outcome_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
