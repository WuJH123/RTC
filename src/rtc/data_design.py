from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .inp import ActuatorCatalog


def canonical_action_sha(settings: dict[str, float]) -> str:
    payload = json.dumps(
        {k: float(settings[k]) for k in sorted(settings)}, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_sequence_sha(sequence: list[dict[str, float]]) -> str:
    payload = [{k: float(step[k]) for k in sorted(step)} for step in sequence]
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def design_independent_actuator_probes(
    checkpoint_settings: pd.DataFrame,
    catalog: ActuatorCatalog,
    *,
    epsilon: float = 0.15,
    include_center: bool = True,
) -> pd.DataFrame:
    """Create D2 same-checkpoint single-actuator counterfactual experiments."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if "checkpoint_id" not in checkpoint_settings.columns:
        raise ValueError("checkpoint_settings requires checkpoint_id")

    setting_columns = {a.actuator_id: f"setting:{a.actuator_id}" for a in catalog.actuators}
    missing = [col for col in setting_columns.values() if col not in checkpoint_settings.columns]
    if missing:
        raise ValueError(f"checkpoint settings missing {len(missing)} actuator columns")

    metadata_columns = [c for c in checkpoint_settings.columns if c not in set(setting_columns.values())]
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


def design_multi_actuator_rollouts(
    checkpoint_settings: pd.DataFrame,
    catalog: ActuatorCatalog,
    *,
    horizon_steps: int,
    sequences_per_checkpoint: int = 8,
    perturbation_std: float = 0.20,
    change_probability: float = 0.25,
    max_delta_per_update: float | None = None,
    seed: int = 42,
    include_hold: bool = True,
) -> pd.DataFrame:
    """Create D3 state-dependent, engineering-feasible continuous action sequences.

    Every discovered actuator remains eligible. ``change_probability`` is sampling coverage,
    not a runtime Top-K. When the frozen controller defines a setting slew/rate limit, the
    same sequential limit is imposed on D3 so Step2 is not trained or ranked primarily on
    action transitions the production MPC is forbidden to execute.
    """

    if horizon_steps <= 0 or sequences_per_checkpoint <= 0:
        raise ValueError("horizon_steps and sequences_per_checkpoint must be positive")
    if perturbation_std <= 0 or not 0.0 < change_probability <= 1.0:
        raise ValueError("invalid perturbation_std/change_probability")
    if max_delta_per_update is not None and float(max_delta_per_update) < 0:
        raise ValueError("max_delta_per_update must be non-negative or None")
    setting_columns = {a.actuator_id: f"setting:{a.actuator_id}" for a in catalog.actuators}
    missing = [c for c in setting_columns.values() if c not in checkpoint_settings.columns]
    if "checkpoint_id" not in checkpoint_settings.columns or missing:
        raise ValueError("checkpoint_id and every setting:<actuator> column are required")
    metadata_columns = [c for c in checkpoint_settings.columns if c not in set(setting_columns.values())]
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    ids = list(catalog.ids)
    lows = np.array([catalog.by_id(a).min_setting for a in ids], dtype=float)
    highs = np.array([catalog.by_id(a).max_setting for a in ids], dtype=float)
    max_delta = None if max_delta_per_update is None else float(max_delta_per_update)

    for _, row in checkpoint_settings.iterrows():
        base = np.array([float(row[setting_columns[a]]) for a in ids], dtype=float)
        count = sequences_per_checkpoint + int(include_hold)
        for seq_idx in range(count):
            current = base.copy()
            sequence: list[dict[str, float]] = []
            if include_hold and seq_idx == 0:
                sequence = [
                    {aid: float(base[i]) for i, aid in enumerate(ids)}
                    for _ in range(horizon_steps)
                ]
                role = "D3_HOLD_REFERENCE"
            else:
                for _step in range(horizon_steps):
                    change = rng.random(len(ids)) < change_probability
                    if not change.any():
                        change[int(rng.integers(0, len(ids)))] = True
                    proposal = current + rng.normal(0.0, perturbation_std, len(ids)) * change
                    if max_delta is not None:
                        proposal = np.minimum(proposal, current + max_delta)
                        proposal = np.maximum(proposal, current - max_delta)
                    current = np.clip(proposal, lows, highs)
                    sequence.append({aid: float(current[i]) for i, aid in enumerate(ids)})
                role = "D3_MULTI_ACTUATOR_ROLLOUT"
            rec: dict[str, object] = {c: row[c] for c in metadata_columns}
            rec.update(
                {
                    "data_role": role,
                    "sequence_index": seq_idx,
                    "horizon_steps": horizon_steps,
                    "sequence_sha256": canonical_sequence_sha(sequence),
                    "settings_sequence_json": json.dumps(sequence, sort_keys=True),
                    "all_actuators_eligible": True,
                    "fixed_active_subset": False,
                    "max_setting_delta_per_update": (
                        "" if max_delta is None else float(max_delta)
                    ),
                    "sequence_rate_feasible": True,
                }
            )
            records.append(rec)
    return pd.DataFrame.from_records(records)


def select_active_learning_cases(
    candidates: pd.DataFrame,
    *,
    budget: int,
    uncertainty_col: str = "model_uncertainty",
    threshold_distance_col: str = "safety_threshold_distance",
    rollout_error_col: str = "rollout_error",
    gradient_disagreement_col: str = "gradient_disagreement",
    rainfall_group_col: str = "rainfall_group",
) -> pd.DataFrame:
    """D4: prioritise uncertain/near-boundary/poor-gradient cases with group diversity."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    required = {
        uncertainty_col,
        threshold_distance_col,
        rollout_error_col,
        gradient_disagreement_col,
        rainfall_group_col,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"active-learning candidates missing columns: {missing}")
    frame = candidates.copy()

    def robust_z(series: pd.Series) -> pd.Series:
        x = series.astype(float)
        med = float(x.median())
        scale = float((x - med).abs().median())
        if scale <= 1e-12:
            scale = float(x.std()) or 1.0
        return (x - med) / scale

    frame["_active_score"] = (
        robust_z(frame[uncertainty_col])
        + robust_z(frame[rollout_error_col])
        + robust_z(frame[gradient_disagreement_col])
        - robust_z(frame[threshold_distance_col].abs())
    )
    frame = frame.sort_values("_active_score", ascending=False)
    selected: list[int] = []
    for _, group in frame.groupby(rainfall_group_col, sort=False):
        if len(selected) >= budget:
            break
        selected.append(int(group.index[0]))
    for idx in frame.index:
        if len(selected) >= budget:
            break
        if int(idx) not in selected:
            selected.append(int(idx))
    return frame.loc[selected].drop(columns=["_active_score"]).reset_index(drop=True)


def summarise_probe_design(manifest: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(manifest)),
        "checkpoints": int(manifest["checkpoint_id"].nunique()),
        "actuators": int(manifest["actuator_id"].nunique()),
        "roles": sorted(manifest["data_role"].unique().tolist()),
        "single_actuator_only": bool(manifest["all_other_actuators_fixed"].all()),
    }
