from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any

import torch

from .step2_train_v127_control import (
    V127ControlTrainingDesign,
    informative_pair_threshold_v127,
)

V128_CONTROL_PROFILE_CONTRACT = (
    "PROJECT7_V128_CONTROL_IDENTIFIABILITY_4060_PROFILE_V1"
)

# SWMM counterfactual branches are deterministic.  The historical V127 default also
# scaled the ranking deadband with the event TFV (0.1% of the reference event), which
# can remove operationally meaningful candidate ordering labels in large flood events.
# V128 keeps the existing 1 m3 absolute numerical floor but removes the event-volume
# proportional deadband for *training*.  Evaluation can still report the stricter V127
# informative-pair metric separately when required for backwards comparison.
V128_RANKING_REFERENCE_FRACTION = 0.0


def build_v128_control_training_design(**kwargs: Any) -> V127ControlTrainingDesign:
    """Return the V127 curriculum with ranking supervision kept at 1 m3 resolution.

    This is deliberately an execution/training profile rather than a silent mutation of
    V127.  It keeps all hydraulic, rollout, objective and memory semantics unchanged,
    while preventing reference TFV magnitude from suppressing action-order supervision.
    """

    if "informative_pair_reference_fraction" in kwargs:
        raise ValueError(
            "V128 fixes informative_pair_reference_fraction; do not override it ad hoc"
        )
    design = V127ControlTrainingDesign(
        informative_pair_reference_fraction=V128_RANKING_REFERENCE_FRACTION,
        **kwargs,
    )
    design.validate()
    return design


def v128_training_pair_threshold_m3(reference_tfv_m3: float) -> float:
    """Expose the effective V128 SWMM action-effect floor for audit/tests."""

    return informative_pair_threshold_v127(
        reference_tfv_m3,
        build_v128_control_training_design(),
    )


def configure_v128_cuda_matmul_precision(
    precision: str | None = None,
) -> dict[str, object]:
    """Configure an explicit, auditable CUDA FP32 matmul policy.

    ``high`` is useful on RTX 4060-class GPUs because PyTorch may use TF32/tensor-core
    paths for FP32 matrix multiplication.  The choice is recorded by the caller and can
    be set to ``highest`` for strict FP32 comparison runs.  AMP remains disabled; this
    function changes only PyTorch's FP32 matmul internal precision policy.
    """

    selected = str(
        precision
        if precision is not None
        else os.environ.get("RTC_V128_MATMUL_PRECISION", "high")
    ).strip().lower()
    if selected not in {"highest", "high", "medium"}:
        raise ValueError(
            "RTC_V128_MATMUL_PRECISION must be one of highest/high/medium"
        )
    before = str(torch.get_float32_matmul_precision())
    torch.set_float32_matmul_precision(selected)
    return {
        "contract": V128_CONTROL_PROFILE_CONTRACT,
        "float32_matmul_precision_before": before,
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "amp_enabled": False,
        "training_design": asdict(build_v128_control_training_design()),
    }


__all__ = [
    "V128_CONTROL_PROFILE_CONTRACT",
    "V128_RANKING_REFERENCE_FRACTION",
    "build_v128_control_training_design",
    "configure_v128_cuda_matmul_precision",
    "v128_training_pair_threshold_m3",
]
