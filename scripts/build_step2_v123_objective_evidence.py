"""Freeze and document the V123 TFV-primary / one-sided PFV-soft objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtc.step3_objective_v123 import TFVPFVObjectiveV123, tfv_pfv_score_v123


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 objective evidence")
    parser.add_argument("--tfv-report", required=True)
    parser.add_argument("--pfv-report", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tfv = json.loads(Path(args.tfv_report).read_text(encoding="utf-8"))
    pfv = json.loads(Path(args.pfv_report).read_text(encoding="utf-8"))
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = cal["calibration"]
    if "pfv_false_safety_margin_m3" not in calibration:
        raise ValueError("V123 objective evidence requires PFV false-safety calibration")
    if "tfv_false_benefit_margin_m3" not in calibration:
        raise ValueError("V123 objective evidence requires TFV false-benefit calibration")

    # The calibrated PFV error budget serves two distinct roles that algebraically
    # cancel for a zero predicted deterioration: it is added to active-command PFV risk
    # and used as the soft deadband.  Therefore a point prediction of DeltaPFV <= 0 is
    # not penalised solely because the model is uncertain, while predicted deterioration
    # is penalised conservatively. PFV remains soft, never a hard feasibility gate.
    pfv_error_margin = float(calibration["pfv_false_safety_margin_m3"])
    contract = TFVPFVObjectiveV123(
        pfv_soft_margin_m3=pfv_error_margin,
        pfv_scale_m3=float(pfv["target_scale_pfv_m3"]),
        tfv_scale_m3=float(tfv["target_scales"]["direct_tfv_scale_m3"]),
        pfv_penalty_weight=0.5,
        pfv_model_error_margin_m3=pfv_error_margin,
    )
    contract.validate()

    probe = tfv_pfv_score_v123(
        torch.tensor([[-1000.0, -800.0]], dtype=torch.float32),
        torch.tensor([[-100.0, 500.0]], dtype=torch.float32),
        movement=torch.tensor([[0.1, 0.1]], dtype=torch.float32),
        contract=contract,
    )
    holdout = tfv["arms"]["B_CAUSAL"]["metrics"]["holdout_d3"]
    continuous_gate = {
        "rank": float(holdout["rank"]) >= 0.70,
        "top1": float(holdout["top1_rate"]) >= 0.50,
        "gradient_sign": False,
        "gradient_cosine": False,
    }
    payload = {
        "contract": "PROJECT7_V123_TFV_PRIMARY_SOFT_PFV_OBJECTIVE_EVIDENCE_V2",
        "objective": {
            "tfv_primary": True,
            "pfv_one_sided_soft": True,
            "pfv_soft_margin_m3": contract.pfv_soft_margin_m3,
            "pfv_model_error_margin_m3": contract.pfv_model_error_margin_m3,
            "pfv_scale_m3": contract.pfv_scale_m3,
            "tfv_scale_m3": contract.tfv_scale_m3,
            "pfv_penalty_weight": contract.pfv_penalty_weight,
            "tfv_false_benefit_margin_m3": float(
                calibration["tfv_false_benefit_margin_m3"]
            ),
            "global_peak": "report_only",
        },
        "semantic_probe": {
            "candidate_tfv_m3": [-1000.0, -800.0],
            "candidate_pfv_m3": [-100.0, 500.0],
            "scores_m3_equivalent": probe["score_m3_equivalent"].tolist(),
            "pfv_risks_m3": probe["pfv_risk_m3"].tolist(),
            "pfv_penalties_m3_equivalent": probe[
                "pfv_penalty_m3_equivalent"
            ].tolist(),
            "pfv_improvement_is_not_negative_reward": True,
            "pfv_deterioration_uses_calibrated_conservative_risk": True,
        },
        "continuous_mpc": {
            "authorized": bool(all(continuous_gate.values())),
            "gate": continuous_gate,
            "reason": "causal Holdout D3 does not meet frozen rank/sign/gradient gate",
        },
        "calibration_source": str(Path(args.calibration).resolve()),
        "boundary": {
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    out.with_suffix(".md").write_text(
        "\n".join(
            [
                "# V123 TFV/PFV objective",
                "",
                f"TFV primary: {payload['objective']['tfv_primary']}",
                f"PFV soft margin: {contract.pfv_soft_margin_m3:.3f} m3",
                f"PFV model-error margin: {contract.pfv_model_error_margin_m3:.3f} m3",
                f"PFV penalty weight: {contract.pfv_penalty_weight}",
                f"Continuous MPC authorized: {payload['continuous_mpc']['authorized']}",
                "",
                "PFV improvement is not a negative reward; active-command PFV deterioration uses calibrated conservative risk.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
