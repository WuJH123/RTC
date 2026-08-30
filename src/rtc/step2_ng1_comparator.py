"""Development-only comparison of NG1 against the frozen V5 frontier."""
from __future__ import annotations

from typing import Any, Mapping


V5_REFERENCE = {
    "d2": {"rank": 0.27214757459444777, "pairwise": 0.6010867089171597, "sign": 0.5958034832015809, "top1_fraction": 0.15625, "selected_harmful_fraction": 0.25, "selected_regret_m3": 6729.969095945358, "delta_tfv_mae_m3": 5820.492691040039},
    "d3": {"rank": 0.38187735361977504, "pairwise": 0.6423964161270062, "sign": 0.614363501082251, "top1_fraction": 0.625, "selected_harmful_fraction": 0.03125, "selected_regret_m3": 4432.721343994141, "delta_tfv_mae_m3": 6686.748023986816},
}


def _evaluation(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, Mapping) or key not in evaluations:
        raise ValueError(f"report lacks evaluations.{key}")
    value = evaluations[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"report evaluations.{key} is not an object")
    return value


def _metric(payload: Mapping[str, Any], key: str, name: str) -> float:
    value = _evaluation(payload, key).get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"report evaluations.{key}.{name} is missing/non-numeric")
    return float(value)


def compare_frontier_reports(v5_report: Mapping[str, Any], candidate_report: Mapping[str, Any]) -> dict[str, Any]:
    """Return all preregistered frontier decisions without changing thresholds."""
    v5: dict[str, dict[str, float]] = {}
    candidate: dict[str, dict[str, float]] = {}
    for label, key in (("d2", "internal_holdout_d2"), ("d3", "internal_holdout_d3")):
        v5[label] = {name: _metric(v5_report, key, name) for name in V5_REFERENCE[label]}
        candidate[label] = {name: _metric(candidate_report, key, name) for name in V5_REFERENCE[label]}
    d2_v5, d2_new = v5["d2"], candidate["d2"]
    d3_v5, d3_new = v5["d3"], candidate["d3"]
    pareto = bool(
        d2_new["rank"] > d2_v5["rank"]
        and d2_new["pairwise"] >= d2_v5["pairwise"]
        and d2_new["selected_harmful_fraction"] <= d2_v5["selected_harmful_fraction"]
        and d2_new["selected_regret_m3"] < d2_v5["selected_regret_m3"]
        and d3_new["rank"] >= d3_v5["rank"]
        and d3_new["pairwise"] >= d3_v5["pairwise"]
        and d3_new["sign"] >= d3_v5["sign"]
        and d3_new["selected_harmful_fraction"] <= d3_v5["selected_harmful_fraction"]
        and d3_new["selected_regret_m3"] <= d3_v5["selected_regret_m3"]
    )
    result: dict[str, Any] = {
        "contract": "PROJECT7_STEP2_NG1_HISTORICAL_FRONTIER_COMPARISON_V1",
        "development_only": True,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_used_for_tuning": False,
        "new_swmm_runs": 0,
        "v5": v5,
        "candidate": candidate,
        "delta_candidate_minus_v5": {
            label: {name: candidate[label][name] - v5[label][name] for name in v5[label]}
            for label in ("d2", "d3")
        },
        "PARETO_CORE_PASS": pareto,
        "CORRECTED_HISTORICAL_FRONTIER": bool(pareto and d2_new["rank"] >= 0.6132),
        "V42_STRETCH_FRONTIER": bool(d2_new["rank"] >= 0.7066),
        "D3_STRETCH_FRONTIER": bool(
            d3_new["rank"] > d3_v5["rank"]
            and d3_new["pairwise"] >= 0.642396
            and d3_new["selected_harmful_fraction"] <= 0.03125
            and d3_new["top1_fraction"] >= 0.65625
            and d3_new["selected_regret_m3"] <= 3007.162
        ),
    }
    return result


__all__ = ["V5_REFERENCE", "compare_frontier_reports"]
