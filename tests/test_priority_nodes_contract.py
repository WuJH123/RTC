from pathlib import Path


EXPECTED_PFV_CORE8 = (
    "MSLBZW001",
    "HS1316314",
    "YS2530050",
    "HS2529198",
    "MH0200773",
    "HS1330349",
    "HS2529139",
    "HS2529052",
)


def test_verified_wuhan_priority_nodes_are_frozen_exactly() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "priority_nodes.txt"
    actual = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert actual == EXPECTED_PFV_CORE8
    assert len(set(actual)) == 8
