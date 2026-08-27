"""Compare primary V27 q95-supported Benchmark5 with V27R1 physical-only ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("event_results"), list):
        raise ValueError(f"invalid Operational Benchmark5 comparison: {path}")
    return value


def _events(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["event_id"]): dict(row) for row in payload["event_results"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v27-supported", required=True)
    parser.add_argument("--v27r1-physical-only", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    supported_path = Path(args.v27_supported).resolve()
    physical_path = Path(args.v27r1_physical_only).resolve()
    supported = _load(supported_path)
    physical = _load(physical_path)
    if supported.get("benchmark_manifest_sha256") != physical.get("benchmark_manifest_sha256"):
        raise ValueError("V27/V27R1 comparison uses different benchmark manifests")
    if supported.get("baseline_cache_sha256") != physical.get("baseline_cache_sha256"):
        raise ValueError("V27/V27R1 comparison uses different baseline caches")
    left = _events(supported)
    right = _events(physical)
    if set(left) != set(right):
        raise ValueError("V27/V27R1 event sets differ")

    rows: list[dict[str, Any]] = []
    supported_sum = physical_sum = 0.0
    reductions: list[float] = []
    physical_wins = 0
    pfv_pass = 0
    for event_id in sorted(left):
        q95 = float(left[event_id]["proposed_tfv_m3"])
        raw = float(right[event_id]["proposed_tfv_m3"])
        supported_sum += q95
        physical_sum += raw
        reduction = 100.0 * (q95 - raw) / q95 if q95 > 0.0 else 0.0
        reductions.append(reduction)
        physical_wins += int(raw < q95 - 1.0e-9)
        pfv_ok = bool(right[event_id].get("proposed_pfv_safety_pass", False))
        pfv_pass += int(pfv_ok)
        rows.append(
            {
                "event_id": event_id,
                "v27_q95_tfv_m3": q95,
                "v27r1_physical_only_tfv_m3": raw,
                "physical_only_minus_q95_tfv_m3": raw - q95,
                "physical_only_reduction_vs_q95_pct": reduction,
                "physical_only_better": raw < q95 - 1.0e-9,
                "v27r1_pfv_safety_pass": pfv_ok,
                "v27r1_flow_routing_error_pct": float(right[event_id].get("flow_routing_error_pct", 0.0)),
            }
        )

    aggregate_reduction = (
        100.0 * (supported_sum - physical_sum) / supported_sum if supported_sum > 0.0 else 0.0
    )
    payload = {
        "contract": "PROJECT7_V27R1_Q95_SUPPORTED_VS_PHYSICAL_ONLY_BENCHMARK5_V1",
        "benchmark_manifest_sha256": supported.get("benchmark_manifest_sha256"),
        "baseline_cache_sha256": supported.get("baseline_cache_sha256"),
        "event_count": len(rows),
        "events_physical_only_better": physical_wins,
        "event_balanced_mean_tfv_reduction_pct": statistics.mean(reductions),
        "event_balanced_median_tfv_reduction_pct": statistics.median(reductions),
        "aggregate_q95_supported_tfv_m3": supported_sum,
        "aggregate_physical_only_tfv_m3": physical_sum,
        "aggregate_physical_only_reduction_vs_q95_pct": aggregate_reduction,
        "pfv_safety_pass_count_physical_only": pfv_pass,
        "pfv_safety_all_events_pass_physical_only": pfv_pass == len(rows),
        "events": rows,
        "interpretation_contract": {
            "positive_reduction_means_physical_only_lower_tfv": True,
            "ablation_is_development_only": True,
            "do_not_promote_based_on_this_file_alone": True,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(out), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
