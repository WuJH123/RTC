from __future__ import annotations

import pandas as pd
import pytest

from rtc.step2_d4_cache_v125 import (
    D4_CANDIDATE_ROLE,
    D4_REFERENCE_ROLE,
    D4_SOURCE_KIND,
    build_d4_run_index_v125,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "plan_row_id": "p0", "d4_split_role": "fit", "event_id": "e1",
            "rainfall_group": "r1", "checkpoint_id": "c1", "sequence_sha256": "s0",
            "candidate_family": "anchor_scale_1.00",
        },
        {
            "plan_row_id": "p1", "d4_split_role": "fit", "event_id": "e1",
            "rainfall_group": "r1", "checkpoint_id": "c1", "sequence_sha256": "s1",
            "candidate_family": "anchor_group_0_plus25",
        },
        {
            "plan_row_id": "p2", "d4_split_role": "audit", "event_id": "e2",
            "rainfall_group": "r2", "checkpoint_id": "c2", "sequence_sha256": "s2",
            "candidate_family": "anchor_scale_1.00",
        },
        {
            "plan_row_id": "p3", "d4_split_role": "audit", "event_id": "e2",
            "rainfall_group": "r2", "checkpoint_id": "c2", "sequence_sha256": "s3",
            "candidate_family": "hold",
        },
    ])


def _runs() -> pd.DataFrame:
    rows = []
    for event, rain, checkpoint, sha in (
        ("e1", "r1", "c1", "s0"), ("e1", "r1", "c1", "s1"),
        ("e2", "r2", "c2", "s2"), ("e2", "r2", "c2", "s3"),
    ):
        rows.append({
            "event_id": event, "rainfall_group": rain, "scientific_split": "development",
            "development_fold": "train", "checkpoint_id": checkpoint,
            "sequence_sha256": sha, "metadata_path": f"{sha}.metadata.json",
            "data_role": "ignored_runner_role", "status": "completed",
        })
    return pd.DataFrame(rows)


def test_d4_fit_run_index_uses_anchor_as_reference_and_source_d4() -> None:
    result = build_d4_run_index_v125(_manifest(), _runs(), split_role="fit")
    assert set(result["source_kind"]) == {D4_SOURCE_KIND}
    assert set(result["data_role"]) == {D4_REFERENCE_ROLE, D4_CANDIDATE_ROLE}
    assert set(result["rainfall_group"]) == {"r1"}
    assert int((result["data_role"] == D4_REFERENCE_ROLE).sum()) == 1


def test_d4_run_index_rejects_rainfall_split_leakage() -> None:
    manifest = _manifest()
    manifest.loc[1, "d4_split_role"] = "audit"
    with pytest.raises(ValueError, match="split leaks"):
        build_d4_run_index_v125(manifest, _runs(), split_role="fit")
