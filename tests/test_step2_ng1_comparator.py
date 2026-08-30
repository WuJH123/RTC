from __future__ import annotations

from rtc.step2_ng1_comparator import compare_frontier_reports


def _metrics(d2: dict[str, float], d3: dict[str, float]) -> dict[str, object]:
    return {"evaluations": {"internal_holdout_d2": d2, "internal_holdout_d3": d3}}


def test_frontier_comparator_reports_pareto_and_historical_frontiers() -> None:
    v5 = _metrics(
        {"rank": 0.27, "pairwise": 0.60, "sign": 0.59, "top1_fraction": 0.15, "selected_harmful_fraction": 0.25, "selected_regret_m3": 6700.0, "delta_tfv_mae_m3": 5800.0},
        {"rank": 0.38, "pairwise": 0.64, "sign": 0.61, "top1_fraction": 0.625, "selected_harmful_fraction": 0.03125, "selected_regret_m3": 4432.0, "delta_tfv_mae_m3": 6686.0},
    )
    candidate = _metrics(
        {"rank": 0.62, "pairwise": 0.65, "sign": 0.64, "top1_fraction": 0.20, "selected_harmful_fraction": 0.20, "selected_regret_m3": 5000.0, "delta_tfv_mae_m3": 5500.0},
        {"rank": 0.40, "pairwise": 0.65, "sign": 0.62, "top1_fraction": 0.66, "selected_harmful_fraction": 0.02, "selected_regret_m3": 3000.0, "delta_tfv_mae_m3": 6500.0},
    )
    result = compare_frontier_reports(v5, candidate)
    assert result["PARETO_CORE_PASS"] is True
    assert result["CORRECTED_HISTORICAL_FRONTIER"] is True
    assert result["V42_STRETCH_FRONTIER"] is False
    assert result["D3_STRETCH_FRONTIER"] is True
