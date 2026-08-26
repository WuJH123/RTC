from __future__ import annotations

from pathlib import Path

import numpy as np

from rtc.project7_v26_historical_supervision import (
    CanonicalCandidateRecord,
    ContextResolver,
    HistoricalCandidateRecord,
    adjudicate_canonical_duplicates,
    canonical_dedup_key,
    canonicalize_record,
    deterministic_split,
    leakage_components,
    read_candidate_records,
    recover_missing_contexts,
)


def _legacy_bank(path: Path, *, groups: int = 6) -> None:
    rows = groups * 2
    node_count = 5
    current = np.zeros((rows, node_count, 4), dtype=np.float32)
    rain = np.zeros((rows, 1, 12, node_count, 1), dtype=np.float32)
    active = np.zeros((rows, 109), dtype=np.float32)
    flow = np.zeros((rows, 109), dtype=np.float32)
    target = np.zeros((rows, 109), dtype=np.float32)
    rainfall_group = []
    event_id = []
    query = []
    source = []
    role = []
    truth = []
    for group in range(groups):
        for candidate in range(2):
            index = 2 * group + candidate
            current[index, :, 0] = float(group)
            rain[index, :, :, :, 0] = float(group) / 10.0
            target[index, candidate] = 0.25 + 0.25 * candidate
            rainfall_group.append(f"rain-{group}")
            event_id.append(f"event-{group}")
            query.append(f"query-{group}")
            source.append(
                "OLD_HYDRAULIC_FAMILY"
                if candidate
                else "STEP2_H10_PROBE_SCALE_0.50"
            )
            role.append(
                "policy_return_calibration" if group % 2 else "policy_return_train"
            )
            truth.append(float(group * 100 - candidate * 25))
    np.savez_compressed(
        path,
        current_state=current,
        rainfall_scenarios=rain,
        active_target=active,
        previous_actuator_flow=flow,
        candidate_target=target,
        true_policy_return_delta_tfv_m3=np.asarray(truth, dtype=np.float64),
        rainfall_group=np.asarray(rainfall_group),
        event_id=np.asarray(event_id),
        query_set_id=np.asarray(query),
        candidate_source=np.asarray(source),
        source_data_role=np.asarray(role),
        continuation_policy_sha256=np.asarray("c" * 64),
    )


def _context(value: float) -> dict[str, np.ndarray]:
    node_count = 5
    state = np.zeros((node_count, 4), dtype=np.float32)
    state[:, 0] = value
    rain = np.zeros((1, 12, node_count, 1), dtype=np.float32)
    rain[..., 0] = value / 10.0
    return {
        "current_state": state,
        "rainfall_scenarios": rain,
        "active_target": np.zeros(109, dtype=np.float32),
        "previous_actuator_flow": np.zeros(109, dtype=np.float32),
    }


def _target(index: int, value: float = 0.5) -> np.ndarray:
    target = np.zeros(109, dtype=np.float32)
    target[index] = value
    return target


def _canonical(
    *,
    context_value: float,
    target_index: int,
    truth: float,
    origin: str,
    derived: bool = False,
    continuation: str = "c" * 64,
) -> CanonicalCandidateRecord:
    row = {
        "rainfall_group": "rain-0",
        "event_id": "event-0",
        "query_set_id": "query-0",
        "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
        "candidate_target": _target(target_index).tolist(),
        "true_policy_return_delta_tfv_m3": truth,
        "continuation_policy_sha256": continuation,
    }
    if derived:
        row.update(
            {
                "historical_supervision_contract": "old-v26",
                "historical_source_path": origin,
                "leakage_group_id": "old-group",
                "split": "train",
            }
        )
    record = HistoricalCandidateRecord(
        row=row,
        source_path=Path(origin),
        source_index=0,
        embedded_context=_context(context_value),
        embedded_target=_target(target_index),
    )
    converted, reason = canonicalize_record(
        record,
        resolver=ContextResolver(study_root=None),
    )
    assert reason == "eligible"
    assert converted is not None
    return converted


def test_legacy_npz_bank_reuses_old_roles_and_old_candidate_family(tmp_path: Path) -> None:
    bank = tmp_path / "STEP3_DEVELOPMENT_TRAIN.npz"
    _legacy_bank(bank)
    historical = read_candidate_records(bank)
    assert len(historical) == 12

    resolver = ContextResolver(study_root=tmp_path)
    canonical = []
    for record in historical:
        converted, reason = canonicalize_record(record, resolver=resolver)
        assert reason == "eligible"
        assert converted is not None
        canonical.append(converted)

    assert {row.row["historical_original_data_role"] for row in canonical} == {
        "policy_return_train",
        "policy_return_calibration",
    }
    assert "OLD_HYDRAULIC_FAMILY" in {
        row.row["candidate_source"] for row in canonical
    }
    assert all(
        row.row["step1_step2_prior_exposure_excludes_training"] is False
        for row in canonical
    )


def test_same_causal_state_keeps_two_actions_but_exact_replay_dedups(tmp_path: Path) -> None:
    bank = tmp_path / "bank.npz"
    _legacy_bank(bank, groups=3)
    resolver = ContextResolver(study_root=tmp_path)
    records = []
    for historical in read_candidate_records(bank):
        converted, _ = canonicalize_record(historical, resolver=resolver)
        assert converted is not None
        records.append(converted)

    first, second = records[0], records[1]
    assert (
        first.row["causal_context_fingerprint_sha256"]
        == second.row["causal_context_fingerprint_sha256"]
    )
    assert (
        first.row["candidate_first_target_sha256"]
        != second.row["candidate_first_target_sha256"]
    )
    assert canonical_dedup_key(first) != canonical_dedup_key(second)


def test_connected_leakage_groups_never_split_same_rain_event_or_context(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank.npz"
    _legacy_bank(bank, groups=6)
    resolver = ContextResolver(study_root=tmp_path)
    records = []
    for historical in read_candidate_records(bank):
        converted, _ = canonicalize_record(historical, resolver=resolver)
        assert converted is not None
        records.append(converted)

    components = leakage_components(records)
    assert len(set(components.values())) == 6
    split = deterministic_split(
        components.values(),
        seed=42,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    assert set(split.values()) == {"train", "validation", "test"}
    for group in range(6):
        assert split[components[2 * group]] == split[components[2 * group + 1]]


def test_path_name_formal_final_does_not_exclude_exact_candidate_truth(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "formal_final_old_version"
    folder.mkdir()
    bank = folder / "historical.npz"
    _legacy_bank(bank, groups=3)
    assert len(read_candidate_records(bank)) == 6


def test_same_query_multiple_context_files_with_same_causal_state_recover_matched_row(
    tmp_path: Path,
) -> None:
    resolver = ContextResolver(study_root=tmp_path)
    peer_a = HistoricalCandidateRecord(
        row={
            "query_set_id": "q",
            "event_id": "event",
            "decision_index": 7,
            "candidate_target": _target(0).tolist(),
        },
        source_path=tmp_path / "a.npz",
        source_index=0,
        embedded_context=_context(3.0),
        embedded_target=_target(0),
    )
    peer_b = HistoricalCandidateRecord(
        row={
            "query_set_id": "q",
            "event_id": "event",
            "decision_index": 7,
            "candidate_target": _target(1).tolist(),
        },
        source_path=tmp_path / "b.npz",
        source_index=0,
        embedded_context=_context(3.0),
        embedded_target=_target(1),
    )
    matched_target = _target(9, 0.75)
    matched = HistoricalCandidateRecord(
        row={
            "query_set_id": "q",
            "rainfall_group": "rain",
            "event_id": "event",
            "decision_index": 7,
            "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
            "candidate_target": matched_target.tolist(),
            "true_policy_return_delta_tfv_m3": -123.0,
            "continuation_policy_sha256": "d" * 64,
        },
        source_path=tmp_path / "matched.json",
        source_index=0,
        embedded_target=matched_target.copy(),
    )

    report = recover_missing_contexts(
        [peer_a, peer_b, matched],
        resolver=resolver,
    )
    assert report["repaired"] == 1
    assert report["ambiguous"] == 0
    assert matched.embedded_context is not None

    converted, reason = canonicalize_record(matched, resolver=resolver)
    assert reason == "eligible"
    assert converted is not None
    assert np.array_equal(converted.target, matched_target)


def test_same_query_genuinely_different_causal_contexts_remain_ambiguous(
    tmp_path: Path,
) -> None:
    resolver = ContextResolver(study_root=tmp_path)
    peers = [
        HistoricalCandidateRecord(
            row={"query_set_id": "q", "candidate_target": _target(index).tolist()},
            source_path=tmp_path / f"peer-{index}.npz",
            source_index=index,
            embedded_context=_context(float(index + 1)),
            embedded_target=_target(index),
        )
        for index in range(2)
    ]
    matched = HistoricalCandidateRecord(
        row={
            "query_set_id": "q",
            "rainfall_group": "rain",
            "candidate_target": _target(9).tolist(),
            "true_policy_return_delta_tfv_m3": -1.0,
        },
        source_path=tmp_path / "matched.json",
        source_index=0,
        embedded_target=_target(9),
    )
    report = recover_missing_contexts(peers + [matched], resolver=resolver)
    assert report["repaired"] == 0
    assert report["ambiguous"] == 1
    assert matched.embedded_context is None


def test_conflicting_derived_copy_does_not_override_independent_swmm_truth() -> None:
    independent = _canonical(
        context_value=4.0,
        target_index=2,
        truth=-100.0,
        origin="raw-truth.npz",
    )
    derived = _canonical(
        context_value=4.0,
        target_index=2,
        truth=-250.0,
        origin="old-v26-records.jsonl",
        derived=True,
    )
    result = adjudicate_canonical_duplicates([independent, derived])
    assert len(result.records) == 1
    assert result.records[0].row["true_policy_return_delta_tfv_m3"] == -100.0
    assert result.report["resolved_derived_disagreement_key_count"] == 1
    assert result.report["derived_disagreement_record_count"] == 1
    assert result.report["unresolved_conflict_key_count"] == 0


def test_genuinely_conflicting_independent_truth_quarantines_only_that_key() -> None:
    conflict_a = _canonical(
        context_value=5.0,
        target_index=3,
        truth=-10.0,
        origin="run-a.npz",
    )
    conflict_b = _canonical(
        context_value=5.0,
        target_index=3,
        truth=20.0,
        origin="run-b.npz",
    )
    unrelated = _canonical(
        context_value=5.0,
        target_index=4,
        truth=-7.0,
        origin="run-c.npz",
    )
    result = adjudicate_canonical_duplicates([conflict_a, conflict_b, unrelated])
    assert len(result.records) == 1
    assert result.records[0].row["candidate_first_target_sha256"] == unrelated.row[
        "candidate_first_target_sha256"
    ]
    assert result.report["unresolved_conflict_key_count"] == 1
    assert result.report["quarantined_conflict_record_count"] == 2
