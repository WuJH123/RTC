from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .d2_eval import join_manifest_runs


def _exact_tfv(metadata_path: str | Path) -> float:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"branch metadata is not a JSON object: {meta_path}")
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"branch lacks exact node statistics: {meta_path}")
    stats = pd.read_csv(meta_path.parent / str(stats_name), compression="infer")
    if "delta_flooding_volume_m3" not in stats.columns:
        raise ValueError(f"node statistics lack flooding-volume truth: {meta_path}")
    values = stats["delta_flooding_volume_m3"].astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite exact flooding volume: {meta_path}")
    return float(np.clip(values, 0.0, None).sum())


def _threshold(reference_tfv: float, *, absolute_m3: float, relative: float) -> float:
    return max(float(absolute_m3), float(relative) * max(float(reference_tfv), 0.0))


def _checkpoint_summary(
    values: pd.DataFrame,
    *,
    reference_mask: pd.Series,
    absolute_m3: float,
    relative: float,
    source_kind: str,
) -> dict[str, object] | None:
    if values.empty or not reference_mask.any():
        return None
    reference_rows = values.loc[reference_mask].drop_duplicates("metadata_path")
    if len(reference_rows) != 1:
        raise ValueError(
            f"{source_kind} checkpoint requires exactly one unique reference branch"
        )
    reference_tfv = float(reference_rows["exact_tfv_m3"].iloc[0])
    unique = values.drop_duplicates("metadata_path")
    best_tfv = float(unique["exact_tfv_m3"].min())
    worst_tfv = float(unique["exact_tfv_m3"].max())
    improvement = reference_tfv - best_tfv
    required = _threshold(
        reference_tfv, absolute_m3=absolute_m3, relative=relative
    )
    reduction_pct = (
        100.0 * improvement / reference_tfv if reference_tfv > 1e-12 else np.nan
    )
    return {
        "reference_tfv_m3": reference_tfv,
        "best_tfv_m3": best_tfv,
        "worst_tfv_m3": worst_tfv,
        "best_improvement_m3": improvement,
        "best_reduction_pct": reduction_pct,
        "action_spread_m3": worst_tfv - best_tfv,
        "meaningful_improvement": bool(improvement >= required),
        "candidate_count": int(len(unique)),
    }


def build_control_leverage_report(
    *,
    d2_manifest: pd.DataFrame,
    d2_run_summary: pd.DataFrame,
    d3_run_summary: pd.DataFrame | None = None,
    meaningful_absolute_m3: float = 1.0,
    meaningful_relative: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Quantify exact-SWMM control leverage before any Step2 training.

    This is intentionally a diagnostic, not another Formal hard gate. It answers the most
    important early question: do actuator changes measurably alter exact future TFV at the
    sampled hydraulic states? If not, generating a very large surrogate dataset is unlikely
    to rescue the control problem.
    """

    if meaningful_absolute_m3 < 0 or meaningful_relative < 0:
        raise ValueError("meaningful-improvement thresholds must be non-negative")

    d2 = join_manifest_runs(d2_manifest, d2_run_summary).copy()
    required_d2 = {
        "checkpoint_id",
        "candidate_action_sha256",
        "base_action_sha256",
        "metadata_path",
        "actuator_id",
    }
    missing = sorted(required_d2 - set(d2.columns))
    if missing:
        raise ValueError(f"D2 leverage input missing columns: {missing}")

    cache: dict[str, float] = {}

    def exact(path: object) -> float:
        value = str(path)
        if value not in cache:
            cache[value] = _exact_tfv(value)
        return cache[value]

    d2["exact_tfv_m3"] = d2["metadata_path"].map(exact)
    detail_rows: list[dict[str, object]] = []
    checkpoint_keys = ["checkpoint_id"]
    for optional in ("event_id", "rainfall_group"):
        if optional in d2.columns:
            checkpoint_keys.append(optional)

    for key_values, group in d2.groupby(checkpoint_keys, sort=False, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        base_sha = group["base_action_sha256"].astype(str).unique().tolist()
        if len(base_sha) != 1:
            raise ValueError("D2 checkpoint has multiple base actions")
        summary = _checkpoint_summary(
            group,
            reference_mask=group["candidate_action_sha256"].astype(str) == base_sha[0],
            absolute_m3=meaningful_absolute_m3,
            relative=meaningful_relative,
            source_kind="D2",
        )
        if summary is None:
            continue
        row = dict(zip(checkpoint_keys, key_values, strict=True))
        row.update({"source_kind": "D2_LOCAL", **summary})
        detail_rows.append(row)

    actuator_effects: list[dict[str, object]] = []
    for actuator_id, group in d2.groupby("actuator_id", sort=False):
        deviations: list[float] = []
        beneficial: list[float] = []
        for _, checkpoint in group.groupby("checkpoint_id", sort=False):
            center = checkpoint[
                checkpoint["candidate_action_sha256"].astype(str)
                == checkpoint["base_action_sha256"].astype(str)
            ].drop_duplicates("metadata_path")
            if len(center) != 1:
                continue
            reference = float(center["exact_tfv_m3"].iloc[0])
            perturbed = checkpoint[
                checkpoint["candidate_action_sha256"].astype(str)
                != checkpoint["base_action_sha256"].astype(str)
            ].drop_duplicates("metadata_path")
            if perturbed.empty:
                continue
            deltas = perturbed["exact_tfv_m3"].astype(float).to_numpy() - reference
            deviations.append(float(np.max(np.abs(deltas))))
            beneficial.append(float(max(0.0, -float(np.min(deltas)))))
        if deviations:
            actuator_effects.append(
                {
                    "actuator_id": str(actuator_id),
                    "checkpoints": len(deviations),
                    "median_abs_tfv_effect_m3": float(np.median(deviations)),
                    "max_abs_tfv_effect_m3": float(np.max(deviations)),
                    "median_best_benefit_m3": float(np.median(beneficial)),
                }
            )

    if d3_run_summary is not None and not d3_run_summary.empty:
        required_d3 = {"checkpoint_id", "data_role", "metadata_path"}
        missing = sorted(required_d3 - set(d3_run_summary.columns))
        if missing:
            raise ValueError(f"D3 leverage input missing columns: {missing}")
        d3 = d3_run_summary.copy()
        d3["exact_tfv_m3"] = d3["metadata_path"].map(exact)
        d3_keys = ["checkpoint_id"]
        for optional in ("event_id", "rainfall_group"):
            if optional in d3.columns:
                d3_keys.append(optional)
        for key_values, group in d3.groupby(d3_keys, sort=False, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            summary = _checkpoint_summary(
                group,
                reference_mask=group["data_role"].astype(str) == "D3_HOLD_REFERENCE",
                absolute_m3=meaningful_absolute_m3,
                relative=meaningful_relative,
                source_kind="D3",
            )
            if summary is None:
                continue
            row = dict(zip(d3_keys, key_values, strict=True))
            row.update({"source_kind": "D3_JOINT", **summary})
            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("control leverage audit produced no checkpoint comparisons")

    summary_by_kind: dict[str, object] = {}
    for source_kind, group in detail.groupby("source_kind", sort=True):
        meaningful_fraction = float(group["meaningful_improvement"].astype(float).mean())
        finite_reduction = group["best_reduction_pct"].astype(float).to_numpy()
        finite_reduction = finite_reduction[np.isfinite(finite_reduction)]
        summary_by_kind[source_kind] = {
            "checkpoints": int(len(group)),
            "meaningful_improvement_fraction": meaningful_fraction,
            "median_best_improvement_m3": float(group["best_improvement_m3"].median()),
            "median_best_reduction_pct": (
                float(np.median(finite_reduction)) if finite_reduction.size else None
            ),
            "median_action_spread_m3": float(group["action_spread_m3"].median()),
            "max_best_improvement_m3": float(group["best_improvement_m3"].max()),
        }

    fractions = [
        float(value["meaningful_improvement_fraction"])
        for value in summary_by_kind.values()
        if isinstance(value, dict)
    ]
    best_fraction = max(fractions, default=0.0)
    if best_fraction >= 0.30:
        interpretation = "PROMISING_CONTROL_LEVERAGE"
        recommendation = "Proceed to Step2 training with expanded but budgeted D2/D3 data."
    elif best_fraction >= 0.10:
        interpretation = "WEAK_OR_STATE_DEPENDENT_CONTROL_LEVERAGE"
        recommendation = (
            "Expand the pilot across more high-depth/high-flood checkpoints before committing to full Step2 generation."
        )
    else:
        interpretation = "LITTLE_MEASURABLE_CONTROL_LEVERAGE_IN_PILOT"
        recommendation = (
            "Do not scale Step2 data yet; first inspect actuator semantics, controllable facilities, checkpoint selection and hydraulic response."
        )

    effect_frame = pd.DataFrame(actuator_effects)
    report: dict[str, object] = {
        "contract": "EXACT_SWMM_CONTROL_LEVERAGE_PILOT_V1",
        "hard_gate": False,
        "meaningful_absolute_m3": float(meaningful_absolute_m3),
        "meaningful_relative": float(meaningful_relative),
        "source_metrics": summary_by_kind,
        "sampled_actuators": int(effect_frame["actuator_id"].nunique()) if not effect_frame.empty else 0,
        "actuators_with_any_measured_effect": (
            int((effect_frame["max_abs_tfv_effect_m3"] > meaningful_absolute_m3).sum())
            if not effect_frame.empty
            else 0
        ),
        "interpretation": interpretation,
        "recommendation": recommendation,
    }
    return detail, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose exact-SWMM control leverage before expensive Step2 training"
    )
    parser.add_argument("--d2-manifest", required=True)
    parser.add_argument("--d2-run-summary", required=True)
    parser.add_argument("--d3-run-summary")
    parser.add_argument("--out", required=True)
    parser.add_argument("--meaningful-absolute-m3", type=float, default=1.0)
    parser.add_argument("--meaningful-relative", type=float, default=0.01)
    args = parser.parse_args()

    detail, report = build_control_leverage_report(
        d2_manifest=pd.read_csv(args.d2_manifest),
        d2_run_summary=pd.read_csv(args.d2_run_summary),
        d3_run_summary=(pd.read_csv(args.d3_run_summary) if args.d3_run_summary else None),
        meaningful_absolute_m3=args.meaningful_absolute_m3,
        meaningful_relative=args.meaningful_relative,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out.with_suffix(".detail.csv")
    detail.to_csv(detail_path, index=False)
    report["detail_csv"] = str(detail_path)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
