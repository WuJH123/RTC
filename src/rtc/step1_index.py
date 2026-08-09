from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "event_id",
    "rainfall_group",
    "scientific_split",
    "development_fold",
    "metadata_path",
}


def _read(path: str | Path, *, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{source} Step1 index source lacks columns: {missing}")
    frame = frame.copy()
    frame["source_role"] = source
    for col in ("event_id", "rainfall_group", "scientific_split", "development_fold"):
        frame[col] = frame[col].fillna("").astype(str)
    if (frame["scientific_split"] != "development").any():
        raise ValueError(f"{source} Step1 source may contain development rows only")
    if not set(frame["development_fold"]).issubset({"train", "validation"}):
        raise ValueError(f"{source} contains invalid development_fold")
    return frame


def build_step1_index(
    *,
    baseline_index_path: str | Path,
    d1_index_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = _read(baseline_index_path, source="BASELINE")
    if "strategy" in baseline.columns:
        allowed = {"no_control", "internal_rtc"}
        bad = sorted(set(baseline["strategy"].astype(str)) - allowed)
        if bad:
            raise ValueError(
                f"Step1 baseline index contains non-state-coverage strategies: {bad}"
            )

    frames = [baseline]
    if d1_index_path:
        d1 = _read(d1_index_path, source="D1_EXPLORATION")
        if (d1["development_fold"] != "train").any():
            raise ValueError("D1 exploration is development/train-only and cannot enter Step1 validation")
        frames.append(d1)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["metadata_path"] = combined["metadata_path"].astype(str)
    if combined["metadata_path"].duplicated().any():
        duplicates = combined.loc[
            combined["metadata_path"].duplicated(keep=False), "metadata_path"
        ].tolist()
        raise ValueError(f"duplicate Step1 trajectory metadata paths: {duplicates[:10]}")
    if (
        combined.groupby("rainfall_group")["development_fold"].nunique() > 1
    ).any():
        raise ValueError("rainfall-group leakage exists across Step1 train/validation folds")

    train = combined[combined["development_fold"] == "train"].copy()
    validation = combined[combined["development_fold"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Step1 requires non-empty development train and validation indexes")
    if set(train["rainfall_group"]) & set(validation["rainfall_group"]):
        raise RuntimeError("Step1 rainfall-group leakage survived validation")

    sort_cols = ["rainfall_group", "event_id", "source_role", "metadata_path"]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    train = train.sort_values(sort_cols).reset_index(drop=True)
    validation = validation.sort_values(sort_cols).reset_index(drop=True)
    return combined, train, validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile unique group-disjoint Step1 train/validation trajectory indexes"
    )
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--d1-index")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    combined, train, validation = build_step1_index(
        baseline_index_path=args.baseline_index,
        d1_index_path=args.d1_index,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined_path = out / "step1_run_index.csv"
    train_path = out / "train_run_index.csv"
    validation_path = out / "validation_run_index.csv"
    combined.to_csv(combined_path, index=False)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    print(
        json.dumps(
            {
                "contract": "STEP1_RUN_INDEX_V1_GROUP_DISJOINT",
                "rows": int(len(combined)),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_rainfall_groups": int(train["rainfall_group"].nunique()),
                "validation_rainfall_groups": int(validation["rainfall_group"].nunique()),
                "combined": str(combined_path),
                "train": str(train_path),
                "validation": str(validation_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
