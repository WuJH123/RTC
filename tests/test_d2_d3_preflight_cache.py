from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rtc.preflight_identity_cache import PreflightIdentityCache
from rtc.simulation_asset_types import (
    d3_identity,
    d3_identity_from_precomputed,
)
from rtc.simulation_assets import (
    SimulationAssetRegistry,
    d2_identity,
    d2_identity_from_precomputed,
    register_d2_metadata,
    register_stamped_d2_metadata_many,
)


def _inp(path: Path, *, end_time: str = "12:00:00") -> Path:
    path.write_text(
        f"""[OPTIONS]
FLOW_UNITS           CMS
START_DATE           01/01/2020
START_TIME           00:00:00
REPORT_START_DATE    01/01/2020
REPORT_START_TIME    00:00:00
END_DATE             01/01/2020
END_TIME             {end_time}
REPORT_END_DATE      01/01/2020
REPORT_END_TIME      {end_time}

[JUNCTIONS]
N1 0 5 0 0 0

[RAINGAGES]
RG1 INTENSITY 0:05 1.0 TIMESERIES TS1

[TIMESERIES]
TS1 01/01/2020 00:55:00 0
TS1 01/01/2020 01:00:00 1
TS1 01/01/2020 01:05:00 0

[END]
""",
        encoding="utf-8",
    )
    return path


def _reference(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    compact = tmp_path / "reference.compact.npz"
    np.savez_compressed(
        compact,
        elapsed_seconds=np.asarray([0, 3600, 7200], dtype=np.int64),
        node_ids=np.asarray(["N1"]),
        state_si=np.asarray(
            [
                [[0, 0, 0, 0, 0, 0]],
                [[1, 1, 0, 1, 1, 1]],
                [[2, 2, 0, 2, 2, 2]],
            ],
            dtype=np.float32,
        ),
        actuator_ids=np.asarray(["P1"]),
        current_setting=np.asarray([[0.0], [0.5], [0.75]], dtype=np.float32),
    )
    metadata = tmp_path / "reference.json"
    metadata.write_text(
        json.dumps({"compact_file": compact.name, "swmm_engine_version": "5.2.4"}),
        encoding="utf-8",
    )
    return metadata


def test_cached_d2_identity_is_byte_semantically_equal_to_legacy(tmp_path: Path) -> None:
    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    cache = PreflightIdentityCache.build(
        event_paths_to_checkpoints={source: (3600,)},
        reference_paths_to_checkpoints={reference: (3600,)},
    )
    event = cache.event(source)
    ref = cache.reference(reference)

    legacy = d2_identity(
        inp_path=source,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=300,
        horizon_seconds=360 * 60,
    )
    cached = d2_identity_from_precomputed(
        physical_network_sha256=event.physical_network_sha256,
        event_prefix_family_sha256=event.event_prefix_family_sha256,
        checkpoint_seconds=3600,
        checkpoint_state_sha256_value=ref.checkpoint_state_sha256(3600),
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=300,
        horizon_seconds=360 * 60,
    )
    assert cached == legacy


def test_cached_d3_identity_is_byte_semantically_equal_to_legacy(tmp_path: Path) -> None:
    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    cache = PreflightIdentityCache.build(
        event_paths_to_checkpoints={source: (3600,)},
        reference_paths_to_checkpoints={reference: (3600,)},
    )
    event = cache.event(source)
    ref = cache.reference(reference)

    legacy = d3_identity(
        inp_path=source,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        sequence_sha256="b" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=300,
        control_block_seconds=600,
        horizon_seconds=7200,
    )
    cached = d3_identity_from_precomputed(
        physical_network_sha256=event.physical_network_sha256,
        event_prefix_family_sha256=event.event_prefix_family_sha256,
        checkpoint_seconds=3600,
        checkpoint_state_sha256_value=ref.checkpoint_state_sha256(3600),
        sequence_sha256="b" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=300,
        control_block_seconds=600,
        horizon_seconds=7200,
    )
    assert cached == legacy


def test_identity_cache_hashes_each_event_and_opens_each_reference_once(
    tmp_path: Path, monkeypatch
) -> None:
    import rtc.preflight_identity_cache as cache_module

    source_a = _inp(tmp_path / "a.inp")
    source_b = _inp(tmp_path / "b.inp")
    reference_a = _reference(tmp_path / "ref_a")
    reference_b = _reference(tmp_path / "ref_b")
    counts = {"physical": 0, "prefix": 0, "endpoint": 0, "npz": 0}

    original_physical = cache_module.physical_contract_sha256
    original_prefix = cache_module.event_prefix_family_sha256
    original_endpoint = cache_module.simulation_available_seconds
    original_np_load = cache_module.np.load

    def physical(*args, **kwargs):
        counts["physical"] += 1
        return original_physical(*args, **kwargs)

    def prefix(*args, **kwargs):
        counts["prefix"] += 1
        return original_prefix(*args, **kwargs)

    def endpoint(*args, **kwargs):
        counts["endpoint"] += 1
        return original_endpoint(*args, **kwargs)

    def np_load(*args, **kwargs):
        counts["npz"] += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(cache_module, "physical_contract_sha256", physical)
    monkeypatch.setattr(cache_module, "event_prefix_family_sha256", prefix)
    monkeypatch.setattr(cache_module, "simulation_available_seconds", endpoint)
    monkeypatch.setattr(cache_module.np, "load", np_load)

    cache = PreflightIdentityCache.build(
        event_paths_to_checkpoints={source_a: (3600, 7200), source_b: (3600,)},
        reference_paths_to_checkpoints={reference_a: (3600, 7200), reference_b: (3600,)},
    )

    assert len(cache.events) == 2
    assert len(cache.references) == 2
    assert counts == {"physical": 2, "prefix": 2, "endpoint": 2, "npz": 2}
    assert len(cache.reference(reference_a).checkpoint_state_sha256_by_elapsed) == 2


def test_registry_snapshot_uses_one_connection_for_repeated_lookups(tmp_path: Path, monkeypatch) -> None:
    registry = SimulationAssetRegistry(tmp_path / "assets")
    calls = 0
    original_connect = registry._connect

    def connect():
        nonlocal calls
        calls += 1
        return original_connect()

    monkeypatch.setattr(registry, "_connect", connect)
    snapshot = registry.preflight_snapshot()
    for index in range(10):
        assert snapshot.lookup_exact(f"missing-{index}") is None
        assert snapshot.lookup_covering("missing-family", horizon_seconds=3600) is None
    assert calls == 1


def test_stamped_d2_registration_does_not_recompute_source_identity(tmp_path: Path, monkeypatch) -> None:
    import rtc.simulation_assets as assets

    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    compact = tmp_path / "branch.compact.npz"
    stats = tmp_path / "branch.stats.csv.gz"
    compact.write_bytes(b"compact")
    stats.write_bytes(b"stats")
    sim_key, family_key, identity = d2_identity(
        inp_path=source,
        reference_metadata_path=reference,
        checkpoint_seconds=3600,
        candidate_action_sha256="a" * 64,
        swmm_engine_version="5.2.4",
        stride_seconds=300,
        horizon_seconds=3600,
    )
    metadata = tmp_path / "branch.json"
    metadata.write_text(
        json.dumps(
            {
                "data_contract": "D2_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED",
                "same_prefix_verification": {"passed": True},
                "simulation_identity_contract": "RTC_SIMULATION_IDENTITY_V1_STATE_ACTION_ENGINE_BOUND",
                "simulation_identity": identity,
                "simulation_identity_sha256": sim_key,
                "simulation_family_sha256": family_key,
                "compact_file": compact.name,
                "node_statistics_file": stats.name,
            }
        ),
        encoding="utf-8",
    )
    registry = SimulationAssetRegistry(tmp_path / "assets")

    def no_legacy_recompute(*_args, **_kwargs):
        raise AssertionError("stamped registration must not recompute source/reference identity")

    monkeypatch.setattr(assets, "d2_identity", no_legacy_recompute)
    register_d2_metadata(registry, metadata)
    assert registry.lookup_exact(sim_key) is not None

    # The batch API must use one SQLite transaction for the same stamped row set.
    calls = 0
    original_connect = registry._connect

    def connect():
        nonlocal calls
        calls += 1
        return original_connect()

    monkeypatch.setattr(registry, "_connect", connect)
    register_stamped_d2_metadata_many(registry, [metadata, metadata])
    assert calls == 1


def test_d2_census_only_never_constructs_swmm_executor(tmp_path: Path, monkeypatch) -> None:
    import sys

    import rtc.d2_runner as runner

    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    manifest = tmp_path / "d2.csv"
    pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "G1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "E1_T60",
                "checkpoint_minutes": 60,
                "candidate_action_sha256": "a" * 64,
                "candidate_settings_json": "{}",
                "trajectory_metadata_path": str(reference),
                "inp_path": str(source),
            }
        ]
    ).to_csv(manifest, index=False)

    def unexpected_executor(*_args, **_kwargs):
        raise AssertionError("census-only must not construct a SWMM executor")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", unexpected_executor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rtc-run-probes",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "out"),
            "--horizon-minutes",
            "60",
            "--stride-seconds",
            "300",
            "--census-only",
        ],
    )
    runner.main()
    assert (tmp_path / "out" / "REQUEST_CENSUS.json").is_file()
    assert not list((tmp_path / "out").glob("*.compact.npz"))


def test_d3_census_only_never_constructs_swmm_executor(tmp_path: Path, monkeypatch) -> None:
    import sys

    import rtc.d3_batch_v2 as runner

    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    manifest = tmp_path / "d3.csv"
    pd.DataFrame(
        [
            {
                "event_id": "E1",
                "rainfall_group": "G1",
                "scientific_split": "development",
                "development_fold": "train",
                "checkpoint_id": "E1_T60",
                "checkpoint_minutes": 60,
                "sequence_sha256": "b" * 64,
                "settings_sequence_json": "[{}]",
                "trajectory_metadata_path": str(reference),
                "inp_path": str(source),
            }
        ]
    ).to_csv(manifest, index=False)

    def unexpected_executor(*_args, **_kwargs):
        raise AssertionError("census-only must not construct a SWMM executor")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", unexpected_executor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rtc-run-d3-batch",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "out"),
            "--control-block-seconds",
            "600",
            "--stride-seconds",
            "300",
            "--census-only",
        ],
    )
    runner.run_d3_batch_main()
    assert (tmp_path / "out" / "REQUEST_CENSUS.json").is_file()
    assert not list((tmp_path / "out").glob("*.compact.npz"))


def test_d2_worker_passes_cached_runtime_sha_to_branch_runner(monkeypatch) -> None:
    import rtc.d2_runner as runner
    from rtc import swmm_data

    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metadata_path="branch.json", flow_routing_error_pct=0.0)

    monkeypatch.setattr(swmm_data, "run_independent_control_branch", fake_runner)
    monkeypatch.setattr(runner, "_stamp", lambda *_args, **_kwargs: "generation")
    job = {
        "runtime_inp": "runtime.inp",
        "runtime_inp_sha256": "r" * 64,
        "checkpoint_minutes": 60,
        "horizon_minutes": 360,
        "candidate_settings_json": "{}",
        "out_dir": "out",
        "branch_id": "branch",
        "candidate_action_sha256": "a" * 64,
        "simulation_identity_sha256": "s" * 64,
        "simulation_family_sha256": "f" * 64,
        "checkpoint_id": "cp",
        "event_id": "event",
        "rainfall_group": "group",
        "scientific_split": "development",
        "development_fold": "train",
        "reference_metadata_path": "reference.json",
        "stride_seconds": 300,
        "debug_raw": False,
        "keep_engine_files": False,
        "snapshot_horizons_minutes": (),
        "endpoint_preflight": {},
    }

    runner._run_job(job)

    assert captured["inp_sha256"] == "r" * 64


def test_d3_worker_passes_cached_runtime_sha_to_branch_runner(monkeypatch) -> None:
    import rtc.d3_batch_v2 as runner
    from rtc import swmm_sequence

    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metadata_path="branch.json", flow_routing_error_pct=0.0)

    monkeypatch.setattr(swmm_sequence, "run_control_sequence_branch", fake_runner)
    monkeypatch.setattr(runner, "_stamp", lambda *_args, **_kwargs: "generation")
    job = {
        "runtime_inp": "runtime.inp",
        "runtime_inp_sha256": "r" * 64,
        "checkpoint_minutes": 60,
        "settings_sequence_json": "[]",
        "control_block_seconds": 600,
        "out_dir": "out",
        "branch_id": "branch",
        "stride_seconds": 300,
        "reference_metadata_path": "reference.json",
        "event_id": "event",
        "rainfall_group": "group",
        "scientific_split": "development",
        "development_fold": "train",
        "checkpoint_id": "cp",
        "data_role": "D3_MULTI_ACTUATOR_ROLLOUT",
        "sequence_sha256": "q" * 64,
        "simulation_identity_sha256": "s" * 64,
        "simulation_family_sha256": "f" * 64,
        "endpoint_preflight": {},
    }

    runner._run(job)

    assert captured["inp_sha256"] == "r" * 64


def test_preflight_cache_fails_closed_when_source_changes(tmp_path: Path) -> None:
    source = _inp(tmp_path / "event.inp")
    reference = _reference(tmp_path)
    cache_path = tmp_path / "PRECHECK_CACHE.json"
    PreflightIdentityCache.build(
        event_paths_to_checkpoints={source: (3600,)},
        reference_paths_to_checkpoints={reference: (3600,)},
        cache_path=cache_path,
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n; changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after cache capture"):
        PreflightIdentityCache.build(
            event_paths_to_checkpoints={source: (3600,)},
            reference_paths_to_checkpoints={reference: (3600,)},
            cache_path=cache_path,
        )
