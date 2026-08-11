from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _invariant(group: pd.DataFrame, column: str, *, default: str = "") -> str:
    if column not in group.columns:
        return default
    values = group[column].fillna("").astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(
            f"D2 manifest is inconsistent within one executed action for {column}: {values}"
        )
    return values[0]


def _series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series([default] * len(frame), index=frame.index, dtype=str)


def _validate_identity_columns(frame: pd.DataFrame, *, source: str) -> None:
    present = [
        column
        for column in ("simulation_identity_sha256", "simulation_family_sha256")
        if column in frame.columns
    ]
    if not present:
        return
    if len(present) != 2:
        raise ValueError(f"{source} must provide both simulation identity and family columns")
    identity = frame["simulation_identity_sha256"].fillna("").astype(str)
    family = frame["simulation_family_sha256"].fillna("").astype(str)
    if (identity == "").any() or (family == "").any():
        raise ValueError(
            f"{source} mixes identity-aware and legacy rows; normalize before Step2"
        )
    if identity.duplicated().any():
        duplicates = identity[identity.duplicated(keep=False)].head().tolist()
        raise ValueError(f"{source} duplicates simulation identities: {duplicates}")


def build_d2_run_index(manifest: pd.DataFrame, run_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one Step2 row per actually executed/reused unique D2 simulation identity."""

    required_manifest = {"candidate_action_sha256", "checkpoint_id"}
    required_runs = {"candidate_action_sha256", "checkpoint_id", "metadata_path"}
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"D2 manifest missing columns: {missing}")
    missing = sorted(required_runs - set(run_summary.columns))
    if missing:
        raise ValueError(f"D2 run summary missing columns: {missing}")
    _validate_identity_columns(run_summary, source="D2 run summary")

    keys = ["candidate_action_sha256", "checkpoint_id"]
    for optional in ("event_id", "checkpoint_minutes"):
        if optional in manifest.columns and optional in run_summary.columns:
            keys.append(optional)

    provenance_rows: list[dict[str, object]] = []
    for key_values, group in manifest.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row: dict[str, object] = dict(zip(keys, key_values, strict=True))
        for column in (
            "event_id",
            "rainfall_group",
            "scientific_split",
            "development_fold",
            "checkpoint_id",
        ):
            if column not in row:
                row[column] = _invariant(group, column)
            else:
                _invariant(group, column, default=str(row[column]))
        row["base_action_sha256"] = _invariant(group, "base_action_sha256")
        row["data_role"] = "D2_SINGLE_ACTUATOR_COUNTERFACTUAL"
        row["manifest_rows_collapsed"] = int(len(group))
        if "actuator_id" in group.columns:
            probe_actuators = sorted(
                set(group["actuator_id"].fillna("").astype(str)) - {""}
            )
        else:
            probe_actuators = []
        row["probe_actuator_ids_json"] = json.dumps(
            probe_actuators, separators=(",", ":")
        )
        provenance_rows.append(row)
    provenance = pd.DataFrame(provenance_rows)

    if run_summary.duplicated(keys).any():
        dup = run_summary.loc[run_summary.duplicated(keys, keep=False), keys]
        raise ValueError(
            "D2 run summary contains duplicate executed branches: "
            f"{dup.head().to_dict('records')}"
        )
    joined = run_summary.merge(
        provenance, on=keys, how="inner", validate="one_to_one", suffixes=("_run", "")
    )
    if len(joined) != len(run_summary):
        missing_runs = run_summary.merge(
            provenance[keys], on=keys, how="left", indicator=True
        )
        missing_runs = missing_runs[missing_runs["_merge"] != "both"]
        raise ValueError(
            f"D2 run summary contains branches absent from manifest: {len(missing_runs)}"
        )

    for column in (
        "event_id",
        "rainfall_group",
        "scientific_split",
        "development_fold",
        "checkpoint_id",
    ):
        run_column = f"{column}_run"
        if run_column in joined.columns:
            left = _series(joined, run_column)
            right = _series(joined, column)
            mismatch = (left != "") & (right != "") & (left != right)
            if mismatch.any():
                raise ValueError(f"D2 manifest/run-summary lineage mismatch in {column}")
            joined[column] = right.where(right != "", left)

    result = pd.DataFrame(
        {
            "event_id": _series(joined, "event_id"),
            "rainfall_group": _series(joined, "rainfall_group"),
            "scientific_split": _series(joined, "scientific_split"),
            "development_fold": _series(joined, "development_fold"),
            "data_role": _series(
                joined, "data_role", "D2_SINGLE_ACTUATOR_COUNTERFACTUAL"
            ),
            "checkpoint_id": _series(joined, "checkpoint_id"),
            "candidate_action_sha256": _series(joined, "candidate_action_sha256"),
            "base_action_sha256": _series(joined, "base_action_sha256"),
            "action_or_sequence_sha256": _series(joined, "candidate_action_sha256"),
            "simulation_identity_sha256": _series(
                joined, "simulation_identity_sha256"
            ),
            "simulation_family_sha256": _series(
                joined, "simulation_family_sha256"
            ),
            "asset_status": _series(joined, "status", "legacy_unindexed"),
            "metadata_path": _series(joined, "metadata_path"),
            "manifest_rows_collapsed": joined["manifest_rows_collapsed"].astype(int),
            "probe_actuator_ids_json": _series(
                joined, "probe_actuator_ids_json", "[]"
            ),
            "source_kind": "D2",
        }
    )
    if result["metadata_path"].duplicated().any():
        raise ValueError("D2 run index still duplicates an executed metadata branch")
    nonempty_identity = result["simulation_identity_sha256"] != ""
    if nonempty_identity.any() and not nonempty_identity.all():
        raise ValueError("D2 Step2 index cannot mix identity-aware and legacy branch rows")
    if nonempty_identity.all() and result["simulation_identity_sha256"].duplicated().any():
        raise ValueError("D2 Step2 index duplicates a simulation identity")
    return result


def standardize_d3_run_index(run_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "rainfall_group",
        "scientific_split",
        "development_fold",
        "checkpoint_id",
        "sequence_sha256",
        "metadata_path",
    }
    missing = sorted(required - set(run_summary.columns))
    if missing:
        raise ValueError(f"D3 run summary missing columns: {missing}")
    _validate_identity_columns(run_summary, source="D3 run summary")
    result = pd.DataFrame(
        {
            "event_id": _series(run_summary, "event_id"),
            "rainfall_group": _series(run_summary, "rainfall_group"),
            "scientific_split": _series(run_summary, "scientific_split"),
            "development_fold": _series(run_summary, "development_fold"),
            "data_role": _series(
                run_summary, "data_role", "D3_MULTI_ACTUATOR_SEQUENCE"
            ),
            "checkpoint_id": _series(run_summary, "checkpoint_id"),
            "candidate_action_sha256": "",
            "base_action_sha256": _series(run_summary, "base_action_sha256"),
            "action_or_sequence_sha256": _series(run_summary, "sequence_sha256"),
            "simulation_identity_sha256": _series(
                run_summary, "simulation_identity_sha256"
            ),
            "simulation_family_sha256": _series(
                run_summary, "simulation_family_sha256"
            ),
            "asset_status": _series(run_summary, "status", "legacy_unindexed"),
            "metadata_path": _series(run_summary, "metadata_path"),
            "manifest_rows_collapsed": 1,
            "probe_actuator_ids_json": "[]",
            "source_kind": "D3",
        }
    )
    if result["metadata_path"].duplicated().any():
        raise ValueError("D3 run summary duplicates metadata branches")
    return result


def build_step2_run_index(
    *,
    d2_manifest: pd.DataFrame,
    d2_run_summary: pd.DataFrame,
    d3_run_summaries: list[pd.DataFrame] | None = None,
    allow_final: bool = False,
) -> pd.DataFrame:
    frames = [build_d2_run_index(d2_manifest, d2_run_summary)]
    for d3 in d3_run_summaries or []:
        frames.append(standardize_d3_run_index(d3))
    result = pd.concat(frames, ignore_index=True)
    if result["metadata_path"].duplicated().any():
        dup = result.loc[
            result["metadata_path"].duplicated(False), ["metadata_path", "source_kind"]
        ]
        raise ValueError(
            f"Step2 index duplicates executed branches: {dup.head().to_dict('records')}"
        )
    identities = result["simulation_identity_sha256"].fillna("").astype(str)
    if (identities != "").any():
        if (identities == "").any():
            raise ValueError(
                "v0.6.8 Step2 index cannot mix simulation-identity-aware and legacy rows; "
                "replay orchestration against the local asset registry to emit normalized summaries"
            )
        if identities.duplicated().any():
            raise ValueError("Step2 index contains duplicate simulation identities")
    if not allow_final and (result["scientific_split"].astype(str) == "final").any():
        raise ValueError(
            "Final rainfall branches are forbidden in the Step2 training/validation index"
        )
    for path in result["metadata_path"].astype(str):
        if not Path(path).is_file():
            raise ValueError(f"Step2 branch metadata is missing: {path}")
    return result.sort_values(
        [
            "scientific_split",
            "development_fold",
            "event_id",
            "checkpoint_id",
            "source_kind",
            "action_or_sequence_sha256",
        ]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one deduplicated, simulation-identity-aware D2+D3 Step2 run index"
    )
    parser.add_argument("--d2-manifest", required=True)
    parser.add_argument("--d2-run-summary", required=True)
    parser.add_argument("--d3-run-summary", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build_step2_run_index(
        d2_manifest=pd.read_csv(args.d2_manifest),
        d2_run_summary=pd.read_csv(args.d2_run_summary),
        d3_run_summaries=[pd.read_csv(path) for path in args.d3_run_summary],
        allow_final=False,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    identity_aware = bool(
        len(result) and (result["simulation_identity_sha256"].astype(str) != "").all()
    )
    print(
        json.dumps(
            {
                "contract": "STEP2_RUN_INDEX_V3_COUNTERFACTUAL_REFERENCE_BOUND",
                "rows": int(len(result)),
                "d2_rows": int((result["source_kind"] == "D2").sum()),
                "d3_rows": int((result["source_kind"] == "D3").sum()),
                "rainfall_groups": int(result["rainfall_group"].nunique()),
                "simulation_identity_bound": identity_aware,
                "unique_simulation_identities": (
                    int(result["simulation_identity_sha256"].nunique())
                    if identity_aware
                    else 0
                ),
                "base_action_reference_rows": int(
                    (result["base_action_sha256"].astype(str) != "").sum()
                ),
                "splits": result.groupby("scientific_split")[
                    "metadata_path"
                ].count().to_dict(),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
