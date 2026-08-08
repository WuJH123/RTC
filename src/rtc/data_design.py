from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .inp import ActuatorCatalog


def canonical_action_sha(settings: dict[str, float]) -> str:
    payload = json.dumps(
        {k: float(settings[k]) for k in sorted(settings)}, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def design_independent_actuator_probes(
    checkpoint_settings: pd.DataFrame,
    catalog: ActuatorCatalog,
    *,
    epsilon: float = 0.15,
    include_center: bool = True,
) -> pd.DataFrame:
    """Create D2 single-actuator counterfactual experiments.

    Every generated branch starts from an explicit checkpoint and changes exactly one
    actuator. This prevents the sequential-pulse contamination that invalidates causal
    facility attribution when later pulses inherit earlier hydraulic disturbances.

    Required columns:
      - checkpoint_id
      - setting:<actuator_id> for every actuator in the catalog

    Extra columns (event/rainfall/split/...) are copied into the output as provenance.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if "checkpoint_id" not in checkpoint_settings.columns:
        raise ValueError("checkpoint_settings requires checkpoint_id")

    setting_columns = {a.actuator_id: f"setting:{a.actuator_id}" for a in catalog.actuators}
    missing = [col for col in setting_columns.values() if col not in checkpoint_settings.columns]
    if missing:
        raise ValueError(f"checkpoint settings missing {len(missing)} actuator columns")

    metadata_columns = [
        c for c in checkpoint_settings.columns if c not in set(setting_columns.values())
    ]
    records: list[dict[str, object]] = []
    for _, row in checkpoint_settings.iterrows():
        base = {aid: float(row[col]) for aid, col in setting_columns.items()}
        base_sha = canonical_action_sha(base)
        for actuator in catalog.actuators:
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
                        "data_role": "D2_INDEPENDENT_ACTUATOR_PROBE",
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
                    }
                )
                records.append(rec)
    return pd.DataFrame.from_records(records)


def summarise_probe_design(manifest: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(manifest)),
        "checkpoints": int(manifest["checkpoint_id"].nunique()),
        "actuators": int(manifest["actuator_id"].nunique()),
        "roles": sorted(manifest["data_role"].unique().tolist()),
        "single_actuator_only": bool(manifest["all_other_actuators_fixed"].all()),
    }
