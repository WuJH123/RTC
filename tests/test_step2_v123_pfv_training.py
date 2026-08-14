from __future__ import annotations

import numpy as np

from rtc.step2_pfv_training_v123 import derive_priority_target_scale_v123


def test_priority_target_scale_uses_nonzero_trainfit_pfv_labels_only() -> None:
    values = np.asarray([0.0, 10.0, -20.0, 30.0, -40.0], dtype=np.float64)
    # q75(abs(nonzero)) = 32.5, with the contract minimum applied only if needed.
    assert derive_priority_target_scale_v123([values], minimum_m3=100.0) == 100.0
    assert derive_priority_target_scale_v123([values], minimum_m3=1.0) == 32.5


def test_priority_target_scale_rejects_empty_or_nonfinite_fit_labels() -> None:
    for values in ([], [np.asarray([0.0, 0.0])], [np.asarray([np.nan, 1.0])]):
        try:
            derive_priority_target_scale_v123(values, minimum_m3=100.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PFV scale input must fail closed")
