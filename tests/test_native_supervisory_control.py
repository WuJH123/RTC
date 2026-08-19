from __future__ import annotations

from pathlib import Path

import numpy as np

from rtc.direct_tfv_sequence_support import (
    changed_facility_support_limit,
    direct_tfv_sequence_geometry,
)
from rtc.native_supervisory_control import (
    NATIVE_SUPERVISORY_CONTROL_CONTRACT,
    derive_native_supervisory_control,
    validate_native_supervisory_control,
)


def _write_testbed(path: Path) -> tuple[str, ...]:
    pumps = [f"P{i:03d}" for i in range(57)]
    orifices = [f"O{i:03d}" for i in range(42)]
    weirs = [f"W{i:03d}" for i in range(10)]
    ids = tuple(pumps + orifices + weirs)
    lines = ["[JUNCTIONS]", "N0 0 1 0 0 0", "N1 0 1 0 0 0", "", "[PUMPS]"]
    lines.extend(f"{value} N0 N1 CURVE" for value in pumps)
    lines.extend(["", "[ORIFICES]"])
    lines.extend(f"{value} N0 N1 SIDE 0 1 NO 0" for value in orifices)
    lines.extend(["", "[WEIRS]"])
    lines.extend(f"{value} N0 N1 TRANSVERSE 0 1 NO 0 0 NO" for value in weirs)
    lines.extend(["", "[CONTROLS]"])
    controlled = pumps + orifices[:16] + weirs[:9]
    for index, value in enumerate(controlled):
        obj = "PUMP" if value.startswith("P") else "ORIFICE" if value.startswith("O") else "WEIR"
        lines.extend(
            [
                f"RULE R{index:03d}",
                "IF NODE N0 DEPTH >= 0.5",
                f"THEN {obj} {value} SETTING = 1",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return ids


def test_native_control_mask_keeps_109_channels_but_enables_82(tmp_path: Path) -> None:
    inp = tmp_path / "testbed.inp"
    ids = _write_testbed(inp)
    payload = derive_native_supervisory_control(inp, actuator_ids=ids)
    mask = validate_native_supervisory_control(payload, actuator_ids=ids)
    assert payload["contract"] == NATIVE_SUPERVISORY_CONTROL_CONTRACT
    assert payload["model_action_channel_count"] == 109
    assert payload["supervisory_control_dimension"] == 82
    assert payload["passive_setting_channel_count"] == 27
    assert payload["controlled_kind_counts"] == {"orifice": 16, "pump": 57, "weir": 9}
    assert int(mask.sum()) == 82
    assert mask.shape == (109,)
    assert payload["step1_retraining_required"] is False
    assert payload["base_step2_retraining_required"] is False


def test_masked_sequence_geometry_ignores_passive_action_channels() -> None:
    reference = np.zeros((24, 109), dtype=float)
    candidate = reference.copy()
    candidate[:2, 0] = 0.4
    candidate[:2, 100] = 0.9
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    geometry = direct_tfv_sequence_geometry(
        candidate,
        reference,
        control_block_steps=2,
        free_control_blocks=12,
        supervisory_mask=mask,
    )
    assert geometry["first_block_l1"] == 0.4
    assert geometry["h120_l1"] == 0.4
    assert geometry["h120_total_variation_l1"] == 0.8


def test_masked_changed_facility_q95_ceiling_respects_control_dimension() -> None:
    payload = {
        "joint_changed_facility_count_q95": 11.2,
        "joint_changed_facility_count_max": 18.0,
        "supervisory_control_dimension": 82,
    }
    assert changed_facility_support_limit(payload, "q95") == 12
