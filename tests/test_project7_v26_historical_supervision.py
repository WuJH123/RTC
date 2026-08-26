from __future__ import annotations

from pathlib import Path

import numpy as np

from rtc.project7_v26_historical_supervision import (
    ContextResolver,
    canonical_dedup_key,
    canonicalize_record,
    deterministic_split,
    leakage_components,
    read_candidate_records,
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
            source.append("OLD_HYDRAULIC_FAMILY" if candidate else "STEP2_H10_PROBE_SCALE_0.50")
            role.append("policy_return_calibration" if group % 2 else "policy_return_train")
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
    assert "OLD_HYDRAULIC_FAMILY" in {row.row["candidate_source"] for row in canonical}
    assert all(row.row["step1_step2_prior_exposure_excludes_training"] is False for row in canonical)


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
    assert first.row["causal_context_fingerprint_sha256"] == second.row["causal_context_fingerprint_sha256"]
    assert first.row["candidate_first_target_sha256"] != second.row["candidate_first_target_sha256"]
    assert canonical_dedup_key(first) != canonical_dedup_key(second)
    assert canonical_dedup_key(first) == canonical_dedup_key(first)


def test_connected_leakage_groups_never_split_same_rain_event_or_context(tmp_path: Path) -> None:
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
    split = deterministic_split(components.values(), seed=42, train_fraction=0.5, validation_fraction=0.25)
    assert set(split.values()) == {"train", "validation", "test"}
    for group in range(6):
        left = split[components[2 * group]]
        right = split[components[2 * group + 1]]
        assert left == right


def test_path_name_formal_final_does_not_exclude_exact_candidate_truth(tmp_path: Path) -> None:
    folder = tmp_path / "formal_final_old_version"
    folder.mkdir()
    bank = folder / "historical.npz"
    _legacy_bank(bank, groups=3)
    assert len(read_candidate_records(bank)) == 6
