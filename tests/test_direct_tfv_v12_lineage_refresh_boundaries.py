from __future__ import annotations

from rtc.direct_tfv_v12_lineage_audit import audit_v12_admission_lineage


def test_lineage_refresh_never_requests_generic_d3_or_new_rainfall() -> None:
    result = audit_v12_admission_lineage(
        {}, step2_checkpoint_sha256="step2", sequence_support_sha256="support"
    )
    assert result["safe_to_reuse_admission"] is False
    assert result["new_rainfall_required_for_lineage_refresh"] is False
    assert result["generic_d3_required_for_lineage_refresh"] is False
