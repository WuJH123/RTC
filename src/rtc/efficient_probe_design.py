from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_design import canonical_action_sha
from .inp import ActuatorCatalog, discover_actuators


def design_budgeted_independent_probes(
    checkpoint_settings: pd.DataFrame,
    catalog: ActuatorCatalog,
    *,
    epsilon: float = 0.15,
    actuators_per_checkpoint: int = 24,
    seed: int = 42,
    include_center: bool = True,
) -> pd.DataFrame:
    """Design information-efficient D2 probes without shrinking the online action space.

    Every candidate action still specifies every discovered actuator. Only the actuator that
    receives a local +/- perturbation is budgeted. A deterministic rotating permutation is
    used across checkpoints so that, once the total probe slots exceed the actuator count,
    every actuator is covered before the design spends heavily on repeated local probes.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if actuators_per_checkpoint <= 0:
        raise ValueError("actuators_per_checkpoint must be positive")
    if "checkpoint_id" not in checkpoint_settings.columns:
        raise ValueError("checkpoint_settings requires checkpoint_id")

    ids = list(catalog.ids)
    if not ids:
        raise ValueError("empty actuator catalog")
    setting_columns = {aid: f"setting:{aid}" for aid in ids}
    missing = [column for column in setting_columns.values() if column not in checkpoint_settings.columns]
    if missing:
        raise ValueError(f"checkpoint settings missing {len(missing)} actuator columns")

    budget = min(int(actuators_per_checkpoint), len(ids))
    metadata_columns = [
        c for c in checkpoint_settings.columns if c not in set(setting_columns.values())
    ]
    rng = np.random.default_rng(seed)
    order = np.arange(len(ids), dtype=int)
    rng.shuffle(order)

    records: list[dict[str, object]] = []
    for checkpoint_index, (_, row) in enumerate(checkpoint_settings.iterrows()):
        base = {aid: float(row[setting_columns[aid]]) for aid in ids}
        base_sha = canonical_action_sha(base)
        start = (checkpoint_index * budget) % len(ids)
        selected_indices = [int(order[(start + j) % len(ids)]) for j in range(budget)]
        for actuator_index in selected_indices:
            actuator = catalog.by_id(ids[actuator_index])
            center = base[actuator.actuator_id]
            requested = [center - epsilon, center + epsilon]
            if include_center:
                requested.insert(1, center)
            settings_to_run: list[float] = []
            for value in requested:
                clipped = float(np.clip(value, actuator.min_setting, actuator.max_setting))
                if not any(abs(clipped - old) <= 1e-12 for old in settings_to_run):
                    settings_to_run.append(clipped)

            for setting in settings_to_run:
                action = dict(base)
                action[actuator.actuator_id] = setting
                rec: dict[str, object] = {c: row[c] for c in metadata_columns}
                rec.update(
                    {
                        "data_role": "D2_BUDGETED_INDEPENDENT_ACTUATOR_PROBE",
                        "actuator_id": actuator.actuator_id,
                        "actuator_kind": actuator.kind,
                        "base_setting": center,
                        "requested_setting": setting,
                        "setting_delta": setting - center,
                        "base_action_sha256": base_sha,
                        "candidate_action_sha256": canonical_action_sha(action),
                        "candidate_settings_json": json.dumps(action, sort_keys=True),
                        "same_checkpoint_required": True,
                        "all_other_actuators_fixed": True,
                        "catalog_actuator_count": len(ids),
                        "probe_budget_per_checkpoint": budget,
                        "probe_sampling_mode": "DETERMINISTIC_ROTATING_ALL_ACTUATOR_COVERAGE_V1",
                    }
                )
                records.append(rec)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("budgeted D2 probe design produced no rows")
    return result


def summarise_budgeted_probe_design(manifest: pd.DataFrame) -> dict[str, object]:
    required = {
        "checkpoint_id",
        "actuator_id",
        "catalog_actuator_count",
        "probe_budget_per_checkpoint",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"budgeted D2 manifest missing columns: {missing}")
    total_actuators = int(manifest["catalog_actuator_count"].astype(int).max())
    covered = int(manifest["actuator_id"].astype(str).nunique())
    per_checkpoint = (
        manifest.groupby("checkpoint_id")["actuator_id"].nunique().astype(float)
    )
    return {
        "contract": "D2_BUDGETED_ROTATING_PROBE_DESIGN_V1",
        "rows": int(len(manifest)),
        "checkpoints": int(manifest["checkpoint_id"].nunique()),
        "catalog_actuators": total_actuators,
        "covered_actuators": covered,
        "actuator_coverage_fraction": float(covered / max(total_actuators, 1)),
        "mean_actuators_per_checkpoint": float(per_checkpoint.mean()),
        "max_actuators_per_checkpoint": int(per_checkpoint.max()),
        "probe_sampling_mode": "DETERMINISTIC_ROTATING_ALL_ACTUATOR_COVERAGE_V1",
        "online_action_space_reduced": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Design budgeted rotating same-prefix D2 probes while preserving all-actuator online MPC"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--actuators-per-checkpoint", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()

    checkpoints = pd.read_csv(args.checkpoints)
    if "scientific_split" not in checkpoints.columns:
        raise ValueError("D2 checkpoint manifest requires scientific_split lineage")
    if (checkpoints["scientific_split"].astype(str) == "final").any():
        raise ValueError("budgeted D2 probe design refuses Final checkpoints before Policy Lock")
    manifest = design_budgeted_independent_probes(
        checkpoints,
        discover_actuators(args.inp),
        epsilon=args.epsilon,
        actuators_per_checkpoint=args.actuators_per_checkpoint,
        seed=args.seed,
        include_center=not args.no_center,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = summarise_budgeted_probe_design(manifest)
    out.with_suffix(out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
