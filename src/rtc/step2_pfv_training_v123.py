"""TrainFit-only PFV scale helpers for Project7 V12.3."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def derive_priority_target_scale_v123(
    fit_delta_pfv_m3: Iterable[np.ndarray], *, minimum_m3: float = 100.0
) -> float:
    """Freeze a robust PFV physical scale from non-zero TrainFit labels only."""
    if float(minimum_m3) <= 0.0 or not np.isfinite(float(minimum_m3)):
        raise ValueError("PFV scale minimum must be finite and positive")
    chunks = [np.asarray(values, dtype=np.float64).reshape(-1) for values in fit_delta_pfv_m3]
    if not chunks:
        raise ValueError("PFV scale requires TrainFit labels")
    values = np.concatenate(chunks)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("PFV scale labels are empty or non-finite")
    nonzero = np.abs(values[np.abs(values) > 1e-9])
    if nonzero.size == 0:
        raise ValueError("PFV scale has no non-zero TrainFit labels")
    return max(float(np.quantile(nonzero, 0.75)), float(minimum_m3))


__all__ = ["derive_priority_target_scale_v123"]
