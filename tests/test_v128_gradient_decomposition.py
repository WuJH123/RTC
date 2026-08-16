from __future__ import annotations

import numpy as np

from rtc.step2_gradient_audit_v128_dev import _gradient_alignment_summary


def test_gradient_alignment_summary_detects_magnitude_collapse() -> None:
    rows = []
    truth = np.asarray([-100.0, -50.0, 25.0, 0.0], dtype=float)
    pred = np.asarray([-1.0, 0.5, -0.25, 0.1], dtype=float)
    for t, p in zip(truth, pred, strict=True):
        rows.append(
            {
                "true_tfv_gradient_m3_per_setting": float(t),
                "predicted_tfv_gradient_m3_per_setting": float(p),
            }
        )
    metrics = _gradient_alignment_summary(rows)
    assert metrics["gradient_cases_nonzero_truth"] == 3
    assert metrics["truth_gradient_negative_cases"] == 2
    assert metrics["truth_gradient_positive_cases"] == 1
    assert metrics["truth_gradient_zero_cases"] == 1
    assert metrics["predicted_gradient_negative_cases"] == 2
    assert metrics["predicted_gradient_positive_cases"] == 2
    assert 0.0 < float(metrics["predicted_to_true_gradient_l2_ratio"]) < 0.05
    assert 0.0 < float(metrics["median_abs_predicted_to_true_gradient_ratio"]) < 0.05


def test_stage_gradient_entrypoint_is_read_only_development_surface() -> None:
    text = open("scripts/audit_step2_gradient_stage_current_dev.py", encoding="utf-8").read()
    assert 'choices=("stage_a", "stage_b0", "objective")' in text
    assert "load_stage_checkpoint_v128" in text
    assert "evaluate_d2_gradient_v128_development" in text
    assert '"validation_accessed": False' in text
    assert '"final_accessed": False' in text
    assert '"formal_accessed": False' in text
    assert "torch.save" not in text
