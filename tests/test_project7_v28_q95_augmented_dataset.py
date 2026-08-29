from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_project7_v28_q95_augmented_dataset import (
    audit_leakage,
    augment_dataset,
    exact_observation_key,
    main,
)


def _row(*, context: str, action: str, truth: float, split: str = "train") -> dict:
    return {
        "causal_context_fingerprint_sha256": context,
        "candidate_first_target_sha256": action,
        "candidate_target": [0.1, 0.2],
        "true_policy_return_delta_tfv_m3": truth,
        "rainfall_group": f"rain-{context[:8]}",
        "event_id": f"event-{context[:8]}",
        "leakage_group_id": f"group-{context[:8]}",
        "split": split,
        "historical_original_data_role": "policy_return_train",
        "historical_source_format": "jsonl",
        "historical_truth_is_independent_observation": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _base_manifest(records: Path, rows: list[dict]) -> dict:
    import hashlib

    return {
        "contract": "BASE",
        "record_count": len(rows),
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "raw_candidate_record_count": len(rows),
        "eligible_before_exact_dedup": len(rows),
        "adjudication": {},
        "context_recovery": {},
        "leakage_audit": audit_leakage(rows),
    }


def _base_with_all_splits(primary: dict) -> list[dict]:
    return [
        primary,
        _row(context="d" * 64, action="e" * 64, truth=0.0, split="validation"),
        _row(context="f" * 64, action="0" * 64, truth=1.0, split="test"),
    ]


def test_same_context_different_action_is_retained(tmp_path: Path) -> None:
    context = "a" * 64
    base = _base_with_all_splits(_row(context=context, action="b" * 64, truth=-1.0))
    targeted = {
        **_row(context=context, action="c" * 64, truth=-2.0),
        "q95_supported_target_sha256": "c" * 64,
        "candidate_target": [0.3, 0.4],
        "event_id": "event-a",
        "rainfall_group": "rain-a",
    }
    records = tmp_path / "base.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_jsonl(records, base)
    _write_jsonl(targeted_path, [targeted])
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_base_manifest(records, base)), encoding="utf-8")

    result = augment_dataset(
        base_manifest_path=manifest_path,
        base_records_path=records,
        targeted_records_path=targeted_path,
        out_dir=tmp_path / "out",
    )

    assert result["record_count"] == 4
    assert result["targeted_truth_reuse"]["included_count"] == 1
    output = Path(result["records"])
    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    keys = {exact_observation_key(row) for row in rows}
    assert len(keys) == 4
    assert sum(key[0] == context for key in keys) == 2
    assert result["leakage_audit"]["passed"] is True


def test_exact_duplicate_is_skipped(tmp_path: Path) -> None:
    context = "a" * 64
    action = "b" * 64
    base = _base_with_all_splits(_row(context=context, action=action, truth=-1.0))
    targeted = {
        **_row(context=context, action=action, truth=-1.0),
        "q95_supported_target_sha256": action,
    }
    records = tmp_path / "base.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_jsonl(records, base)
    _write_jsonl(targeted_path, [targeted])
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_base_manifest(records, base)), encoding="utf-8")

    result = augment_dataset(
        base_manifest_path=manifest_path,
        base_records_path=records,
        targeted_records_path=targeted_path,
        out_dir=tmp_path / "out",
    )

    assert result["record_count"] == 3
    assert result["targeted_truth_reuse"]["duplicate_truth_skipped_count"] == 1


def test_conflicting_exact_truth_is_quarantined(tmp_path: Path) -> None:
    context = "a" * 64
    action = "b" * 64
    base = _base_with_all_splits(_row(context=context, action=action, truth=-1.0))
    targeted = {
        **_row(context=context, action=action, truth=2.0),
        "q95_supported_target_sha256": action,
    }
    records = tmp_path / "base.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_jsonl(records, base)
    _write_jsonl(targeted_path, [targeted])
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_base_manifest(records, base)), encoding="utf-8")

    result = augment_dataset(
        base_manifest_path=manifest_path,
        base_records_path=records,
        targeted_records_path=targeted_path,
        out_dir=tmp_path / "out",
    )

    assert result["record_count"] == 3
    assert result["adjudication"]["unresolved_conflict_key_count"] == 1
    assert result["rejected_counts"]["unresolved_independent_truth_conflict"] == 1


def test_leakage_audit_catches_cross_split_identity() -> None:
    context = "a" * 64
    rows = [
        _row(context=context, action="b" * 64, truth=0.0, split="train"),
        _row(context=context, action="c" * 64, truth=0.0, split="test"),
    ]
    assert audit_leakage(rows)["passed"] is False


def test_nonempty_output_is_not_overwritten(tmp_path: Path) -> None:
    context = "a" * 64
    base = _base_with_all_splits(_row(context=context, action="b" * 64, truth=-1.0))
    targeted = {
        **_row(context=context, action="c" * 64, truth=-2.0),
        "q95_supported_target_sha256": "c" * 64,
    }
    records = tmp_path / "base.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_jsonl(records, base)
    _write_jsonl(targeted_path, [targeted])
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_base_manifest(records, base)), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        augment_dataset(
            base_manifest_path=manifest_path,
            base_records_path=records,
            targeted_records_path=targeted_path,
            out_dir=out,
        )


def test_cli_entrypoint_maps_path_arguments(tmp_path: Path, monkeypatch) -> None:
    context = "a" * 64
    base = _base_with_all_splits(_row(context=context, action="b" * 64, truth=-1.0))
    targeted = {
        **_row(context=context, action="c" * 64, truth=-2.0),
        "q95_supported_target_sha256": "c" * 64,
    }
    records = tmp_path / "base.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_jsonl(records, base)
    _write_jsonl(targeted_path, [targeted])
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_base_manifest(records, base)), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_project7_v28_q95_augmented_dataset.py",
            "--base-manifest",
            str(manifest_path),
            "--base-records",
            str(records),
            "--targeted-records",
            str(targeted_path),
            "--out-dir",
            str(out),
        ],
    )
    main()
    assert (out / "V28_Q95_MATCHED_AUGMENTED_EXACT_RETURN_DATASET_MANIFEST.json").is_file()
