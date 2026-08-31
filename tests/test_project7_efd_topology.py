from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtc.project7_efd_topology import EFD_TOPOLOGY_CONTRACT, build_efd_topology_map


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "network.inp"
    path.write_text(text, encoding="utf-8")
    return path


def test_efd_maps_storage_through_passive_conduit_to_first_supervisory_outlet(tmp_path: Path) -> None:
    inp = _write(
        tmp_path,
        """
[OPTIONS]
FLOW_UNITS CMS
[STORAGE]
S1 0 4 0 FUNCTIONAL 10 0 0
S2 0 4 0 FUNCTIONAL 10 0 0
[JUNCTIONS]
J1 0 4 0 0 0
J2 0 4 0 0 0
D1 0 4 0 0 0
D2 0 4 0 0 0
[CONDUITS]
C1 S1 J1 1 0.01 0 0 0 0
C2 S2 J2 1 0.01 0 0 0 0
[ORIFICES]
O1 J1 D1 SIDE 0 0.6 NO
O2 J2 D2 SIDE 0 0.6 NO
""".strip()
        + "\n",
    )
    result = build_efd_topology_map(
        inp,
        actuator_ids=("O1", "O2"),
        supervisory_mask=np.asarray([True, True]),
    )
    assert result.contract == EFD_TOPOLOGY_CONTRACT
    assert result.storage_to_actuators == {"S1": ("O1",), "S2": ("O2",)}
    assert result.mapped_storage_ids == ("S1", "S2")
    assert result.mapped_actuator_ids == ("O1", "O2")
    assert result.ambiguous_actuator_ids == ()


def test_efd_traverses_frozen_regulator_as_passive_link(tmp_path: Path) -> None:
    inp = _write(
        tmp_path,
        """
[OPTIONS]
FLOW_UNITS CMS
[STORAGE]
S1 0 4 0 FUNCTIONAL 10 0 0
S2 0 4 0 FUNCTIONAL 10 0 0
[JUNCTIONS]
J1 0 4 0 0 0
J2 0 4 0 0 0
D1 0 4 0 0 0
D2 0 4 0 0 0
[ORIFICES]
FROZEN S1 J1 SIDE 0 0.6 NO
O1 J1 D1 SIDE 0 0.6 NO
O2 S2 D2 SIDE 0 0.6 NO
""".strip()
        + "\n",
    )
    result = build_efd_topology_map(
        inp,
        actuator_ids=("FROZEN", "O1", "O2"),
        supervisory_mask=np.asarray([False, True, True]),
    )
    assert result.storage_to_actuators["S1"] == ("O1",)
    assert result.storage_to_actuators["S2"] == ("O2",)
    assert "FROZEN" not in result.mapped_actuator_ids


def test_efd_stops_at_another_storage_instead_of_stealing_its_outlet(tmp_path: Path) -> None:
    inp = _write(
        tmp_path,
        """
[OPTIONS]
FLOW_UNITS CMS
[STORAGE]
S1 0 4 0 FUNCTIONAL 10 0 0
S2 0 4 0 FUNCTIONAL 10 0 0
S3 0 4 0 FUNCTIONAL 10 0 0
[JUNCTIONS]
D2 0 4 0 0 0
D3 0 4 0 0 0
[CONDUITS]
C12 S1 S2 1 0.01 0 0 0 0
[ORIFICES]
O2 S2 D2 SIDE 0 0.6 NO
O3 S3 D3 SIDE 0 0.6 NO
""".strip()
        + "\n",
    )
    result = build_efd_topology_map(
        inp,
        actuator_ids=("O2", "O3"),
        supervisory_mask=np.asarray([True, True]),
    )
    assert "S1" in result.unmapped_storage_ids
    assert result.storage_to_actuators["S2"] == ("O2",)
    assert result.storage_to_actuators["S3"] == ("O3",)


def test_efd_fails_closed_when_one_supervisory_actuator_is_shared_by_storages(tmp_path: Path) -> None:
    inp = _write(
        tmp_path,
        """
[OPTIONS]
FLOW_UNITS CMS
[STORAGE]
S1 0 4 0 FUNCTIONAL 10 0 0
S2 0 4 0 FUNCTIONAL 10 0 0
[JUNCTIONS]
J 0 4 0 0 0
D 0 4 0 0 0
[CONDUITS]
C1 S1 J 1 0.01 0 0 0 0
C2 S2 J 1 0.01 0 0 0 0
[ORIFICES]
O1 J D SIDE 0 0.6 NO
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="independently controllable storages"):
        build_efd_topology_map(
            inp,
            actuator_ids=("O1",),
            supervisory_mask=np.asarray([True]),
        )


def test_efd_rejects_runtime_actuator_not_present_in_source_inp(tmp_path: Path) -> None:
    inp = _write(
        tmp_path,
        """
[OPTIONS]
FLOW_UNITS CMS
[STORAGE]
S1 0 4 0 FUNCTIONAL 10 0 0
S2 0 4 0 FUNCTIONAL 10 0 0
[JUNCTIONS]
D1 0 4 0 0 0
D2 0 4 0 0 0
[ORIFICES]
O1 S1 D1 SIDE 0 0.6 NO
O2 S2 D2 SIDE 0 0.6 NO
""".strip()
        + "\n",
    )
    with pytest.raises(ValueError, match="absent from the source INP"):
        build_efd_topology_map(
            inp,
            actuator_ids=("O1", "MISSING"),
            supervisory_mask=np.asarray([True, True]),
        )
