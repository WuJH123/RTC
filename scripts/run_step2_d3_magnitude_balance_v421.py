"""Run the bounded V4.2.1 magnitude-balance correctness experiment.

The model/training stages are inherited from the isolated V4.2 runner, but all
outputs are written to a new V4.2.1 directory.  The only changed semantics are
the shared per-group magnitude partition/weights and action-contract scaling.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_step2_d3_magnitude_calibration_v42 import (  # noqa: E402
    STAGES,
    STUDY,
    run_stage,
)
from rtc.step2_train_response_v41 import (  # noqa: E402
    CounterfactualDeltaScalesV41,
    balanced_magnitude_stratum_weights,
    magnitude_strata_metrics_v41,
)


V421_OUT = STUDY / "step2_d3_magnitude_balance_v421"
V42_OLD_STAGE = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "stage_result.json"
V41_OLD_STAGE = STUDY / "step2_response_calibration_v41" / "05_12_group_micro" / "stage_result.json"
AUDIT_V42 = STUDY / "step2_d3_magnitude_pathway_audit_v42" / "STEP2_D3_MAGNITUDE_PATHWAY_AUDIT_V42.json"
SCALE_JSON = STUDY / "step2_response_calibration_v41" / "00_scales" / "counterfactual_delta_scales_train18.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result = {
        name: float(np.nanmean([float(row[name]) for row in rows]))
        for name in ("spread_ratio", "rank", "pairwise", "sign")
    }
    result.update(
        {
            "top1": int(sum(bool(row["top1"]) for row in rows)),
            "groups": len(rows),
            "mean_regret_m3": float(np.mean([float(row["regret_m3"]) for row in rows])),
            "max_regret_m3": float(max(float(row["regret_m3"]) for row in rows)),
        }
    )
    return result


def _stage_reporting(path: Path, scales: CounterfactualDeltaScalesV41) -> dict[str, Any]:
    stage = _load(path)
    contributions = stage.get("candidate_contributions") or stage["training"].get("candidate_contributions", [])
    metrics = stage.get("group_metrics") or stage["training"].get("group_metrics", [])
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    d3 = [row for row in metrics if row["source_kind"] == "D3"]
    return {
        "source": str(path),
        "d3": _aggregate(d3),
        "d3_magnitude_strata_corrected": magnitude_strata_metrics_v41(
            contributions, q33=float(q["q33"]), q67=float(q["q67"])
        ),
    }


def _weight_audit(scales: CounterfactualDeltaScalesV41) -> dict[str, Any]:
    audit = _load(AUDIT_V42)
    rows = [row for row in audit["candidate_rows"] if row["source_kind"] == "D3"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    q = scales.by_source["D3"].tfv_abs_quantiles_m3
    group_audits = []
    for group, group_rows in sorted(grouped.items()):
        truth = torch.tensor(
            [[float(row["true_delta_tfv_m3"]) for row in group_rows]], dtype=torch.float64
        )
        partition = balanced_magnitude_stratum_weights(
            truth, q33=float(q["q33"]), q67=float(q["q67"])
        )
        totals = {
            name: float(
                (partition["weights"] * partition[f"{name}_mask"]).sum().item()
            )
            for name in ("small", "medium", "large")
        }
        counts = {
            name: int(partition[f"{name}_count"][0].item())
            for name in ("small", "medium", "large")
        }
        present_totals = [value for name, value in totals.items() if counts[name] > 0]
        group_audits.append(
            {
                "group": group,
                "candidate_count": len(group_rows),
                "small_count": counts["small"],
                "medium_count": counts["medium"],
                "large_count": counts["large"],
                "small_total_weight": totals["small"],
                "medium_total_weight": totals["medium"],
                "large_total_weight": totals["large"],
                "partition_valid": bool(partition["partition_valid"][0].item()),
                "balanced_valid": bool(partition["balanced_valid"][0].item()),
                "present_strata_equal": bool(
                    max(present_totals) - min(present_totals) <= 1e-6
                ),
            }
        )
    total_counts = {
        name: int(sum(item[f"{name}_count"] for item in group_audits))
        for name in ("small", "medium", "large")
    }
    return {
        "contract": "MAGNITUDE_STRATUM_WEIGHT_AUDIT_V421",
        "q33_m3": float(q["q33"]),
        "q67_m3": float(q["q67"]),
        "frozen_d3_candidate_count": len(rows),
        "group_count": len(group_audits),
        "total_counts": total_counts,
        "partition_valid": all(item["partition_valid"] for item in group_audits)
        and sum(total_counts.values()) == len(rows),
        "balanced_valid": all(
            item["balanced_valid"] and item["present_strata_equal"] for item in group_audits
        ),
        "groups": group_audits,
    }


def main() -> int:
    scales = CounterfactualDeltaScalesV41.from_json_dict(_load(SCALE_JSON))
    for stage in ("tiny-d3", "tiny-combined", "micro"):
        run_stage(stage, device_name="cuda", output_root=V421_OUT)
    weight_audit = _weight_audit(scales)
    corrected_reporting = {
        "contract": "STEP2_D3_MAGNITUDE_BALANCE_CORRECTED_REPORTING_V421",
        "old_v41": _stage_reporting(V41_OLD_STAGE, scales),
        "old_v42": _stage_reporting(V42_OLD_STAGE, scales),
        "new_v421": _stage_reporting(V421_OUT / "04_12_group_micro" / "stage_result.json", scales),
        "weight_audit": weight_audit,
    }
    V421_OUT.mkdir(parents=True, exist_ok=True)
    (V421_OUT / "MAGNITUDE_STRATUM_WEIGHT_AUDIT_V421.json").write_text(
        json.dumps(weight_audit, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (V421_OUT / "CORRECTED_REPORTING_METRICS_V421.json").write_text(
        json.dumps(corrected_reporting, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(V421_OUT),
        "weight_audit": {
            "counts": weight_audit["total_counts"],
            "partition_valid": weight_audit["partition_valid"],
            "balanced_valid": weight_audit["balanced_valid"],
        },
        "corrected_reporting": corrected_reporting,
    }, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
