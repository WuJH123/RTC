from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


CONTRACT = "WUHAN_RTC_V069_INPUT_ADOPTION_18_6_6_SPLIT_LOCK_V2"
NETWORK_NAME = "wuhan_method_testbed_v067.inp"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_bundle_root(input_root: str | Path) -> Path:
    requested = Path(input_root).expanduser().resolve()
    candidates: list[Path] = []
    for root in (requested, *[p for p in requested.iterdir() if p.is_dir()]):
        if (
            (root / "network" / NETWORK_NAME).is_file()
            and (root / "events").is_dir()
            and (root / "rainfall").is_dir()
            and (root / "contracts").is_dir()
        ):
            candidates.append(root)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise ValueError(
            "expected exactly one v0.6.7 bundle root containing network/events/rainfall/contracts "
            f"under {requested}; found {[str(x) for x in unique]}"
        )
    return unique[0]


def _git_head(repo: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNRESOLVED"
    return value


def _verify_active_split(registry: pd.DataFrame, split_contract: dict[str, object]) -> None:
    if len(registry) != 30 or registry["rainfall_group"].astype(str).nunique() != 30:
        raise ValueError("active v0.6.9 registry must contain exactly 30 rainfall groups")
    allowed = {"development", "final"}
    present = set(registry["scientific_split"].astype(str))
    if not present.issubset(allowed) or present != allowed:
        raise ValueError(f"active split must contain only development/final, got {sorted(present)}")
    dev = registry[registry["scientific_split"].astype(str) == "development"].copy()
    final = registry[registry["scientific_split"].astype(str) == "final"].copy()
    train = dev[dev["development_fold"].astype(str) == "train"]
    validation = dev[dev["development_fold"].astype(str) == "validation"]
    if (len(train), len(validation), len(final)) != (18, 6, 6):
        raise ValueError(
            "active Project7 split must be exactly 18 Train / 6 Validation / 6 Final"
        )
    if (final["development_fold"].astype(str) != "").any():
        raise ValueError("Final rows must not carry a development_fold")
    if set(dev["development_fold"].astype(str)) != {"train", "validation"}:
        raise ValueError("development rows must be exactly train/validation")
    for cohort, name, expected_per_duration in (
        (train, "Train", 3),
        (validation, "Validation", 1),
        (final, "Final", 1),
    ):
        counts = cohort.groupby("duration_minutes")["event_id"].count().to_dict()
        if set(int(x) for x in counts) != {60, 120, 180, 240, 300, 360}:
            raise ValueError(f"{name} does not cover every frozen duration")
        if any(int(value) != expected_per_duration for value in counts.values()):
            raise ValueError(f"{name} duration allocation differs from frozen split contract")
    for cohort, name in ((validation, "Validation"), (final, "Final")):
        if set(cohort["return_period_year"].astype(int)) != {5, 10, 20, 50, 100}:
            raise ValueError(f"{name} must span all five return periods")

    expected_counts = split_contract.get("counts")
    if not isinstance(expected_counts, dict):
        raise ValueError("split contract lacks counts")
    if {
        "development_train": int(expected_counts.get("development_train", -1)),
        "development_validation": int(expected_counts.get("development_validation", -1)),
        "final": int(expected_counts.get("final", -1)),
    } != {"development_train": 18, "development_validation": 6, "final": 6}:
        raise ValueError("split contract itself is not the frozen 18/6/6 contract")


def adopt_inputs(
    *,
    input_root: str | Path,
    portable_registry: str | Path | None = None,
    method_contract: str | Path | None = None,
    split_contract: str | Path | None = None,
) -> dict[str, object]:
    repo = _repo_root()
    bundle = _resolve_bundle_root(input_root)
    portable = Path(
        portable_registry
        or repo / "configs" / "project7_v069_events_with_splits.csv"
    ).resolve()
    method = Path(
        method_contract
        or repo
        / "data"
        / "method_testbed_v067"
        / "contracts"
        / "method_testbed_contract.v067.json"
    ).resolve()
    split_path = Path(
        split_contract or repo / "configs" / "project7_v069_split_contract.json"
    ).resolve()
    if not portable.is_file() or not method.is_file() or not split_path.is_file():
        raise ValueError("active GitHub registry/method/split contract is missing from the local repo")

    contract = json.loads(method.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_payload, dict) or split_payload.get("contract") != (
        "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
    ):
        raise ValueError("unexpected active Project7 split contract")
    expected_registry_sha = str(split_payload.get("portable_registry_sha256", ""))
    actual_registry_sha = _sha256(portable)
    if not expected_registry_sha or actual_registry_sha != expected_registry_sha:
        raise ValueError(
            "active portable registry SHA differs from the preregistered split contract"
        )
    registry = pd.read_csv(portable, keep_default_na=False)
    _verify_active_split(registry, split_payload)

    network = bundle / "network" / NETWORK_NAME
    network_sha = _sha256(network)
    expected_network_sha = str(contract["network_sha256"])
    if network_sha != expected_network_sha:
        raise ValueError(
            f"corrected network SHA mismatch: {network_sha} != {expected_network_sha}"
        )

    expected_sensor_sha = str(contract["sensor_layout_sha256"])
    expected_priority_sha = str(contract["priority_nodes_sha256"])
    sensors = bundle / "contracts" / "sensor_nodes.txt"
    priority = bundle / "contracts" / "priority_nodes.txt"
    if _sha256(sensors) != expected_sensor_sha:
        raise ValueError("sensor_nodes.txt differs from the GitHub-pinned v0.6.7 contract")
    if _sha256(priority) != expected_priority_sha:
        raise ValueError("priority_nodes.txt differs from the GitHub-pinned v0.6.7 contract")

    local = registry.copy()
    verified_events = 0
    verified_rainfall = 0
    for idx, row in local.iterrows():
        event_id = str(row["event_id"])
        event = bundle / "events" / f"{event_id}.inp"
        rainfall = bundle / "rainfall" / f"{event_id}.csv"
        if not event.is_file() or not rainfall.is_file():
            raise ValueError(f"missing event/rainfall source for {event_id}")
        event_sha = _sha256(event)
        rain_sha = _sha256(rainfall)
        if event_sha != str(row["prepared_inp_sha256"]):
            raise ValueError(f"event INP SHA mismatch for {event_id}")
        if rain_sha != str(row["rainfall_sha256"]):
            raise ValueError(f"rainfall SHA mismatch for {event_id}")
        local.at[idx, "inp_path"] = str(event.resolve())
        local.at[idx, "rainfall_csv_path"] = str(rainfall.resolve())
        verified_events += 1
        verified_rainfall += 1

    contracts = bundle / "contracts"
    active_registry = contracts / "events_with_splits.csv"
    local.to_csv(active_registry, index=False)
    source_columns = [
        c for c in local.columns if c not in {"scientific_split", "development_fold"}
    ]
    active_source = contracts / "source_event_manifest.csv"
    local[source_columns].to_csv(active_source, index=False)

    split_counts = {
        str(k): int(v) for k, v in local.groupby("scientific_split").size().items()
    }
    dev = local[local["scientific_split"].astype(str) == "development"]
    dev_counts = {
        str(k): int(v) for k, v in dev.groupby("development_fold").size().items()
    }
    return {
        "contract": CONTRACT,
        "passed": True,
        "repo_head": _git_head(repo),
        "resolved_input_root": str(bundle),
        "network_path": str(network),
        "network_sha256": network_sha,
        "events_verified": verified_events,
        "rainfall_files_verified": verified_rainfall,
        "portable_registry": str(portable),
        "portable_registry_sha256": actual_registry_sha,
        "split_contract": str(split_path),
        "split_contract_sha256": _sha256(split_path),
        "active_registry": str(active_registry),
        "active_registry_sha256": _sha256(active_registry),
        "source_event_manifest": str(active_source),
        "source_event_manifest_sha256": _sha256(active_source),
        "sensor_layout_sha256": _sha256(sensors),
        "priority_nodes_sha256": _sha256(priority),
        "scientific_split_counts": split_counts,
        "development_fold_counts": dev_counts,
        "historical_derived_data_reused": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the extracted v0.6.7 physical/rainfall bundle and overwrite its local active "
            "registry with the preregistered v0.6.9 18/6/6 split"
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--portable-registry")
    parser.add_argument("--method-contract")
    parser.add_argument("--split-contract")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = adopt_inputs(
        input_root=args.input_root,
        portable_registry=args.portable_registry,
        method_contract=args.method_contract,
        split_contract=args.split_contract,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
