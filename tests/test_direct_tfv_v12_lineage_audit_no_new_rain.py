from __future__ import annotations

from rtc.direct_tfv_v12_lineage_audit import audit_v12_admission_lineage


def test_refresh_does_not_request_new_rain_or_generic_d3() -> None:
    result = audit_v12_admission_lineage(
        {}, step2_checkpoint_sha256="x", sequence_support_sha256="y"
    )
    assert result["new_rainfall_required_for_lineage_refresh"] is False
    assert result["generic_d3_required_for_lineage_refresh"] is False
    assert result["authoritative_refresh_branch_count"] == 48
