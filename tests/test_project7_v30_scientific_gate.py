from __future__ import annotations

from rtc.project7_v30_scientific_gate import (
    ScientificPanelRow,
    V30_DEVELOPMENT_PARTITION,
    evaluate_v30_scientific_gate,
)


SHA = "a" * 64


def row(
    event: str,
    strategy: str,
    *,
    tfv: float,
    pfv: float,
    actions: int = 1,
    changed: int = 3,
    partition: str = V30_DEVELOPMENT_PARTITION,
    engineering: bool = True,
) -> ScientificPanelRow:
    return ScientificPanelRow(
        event_id=event,
        strategy=strategy,
        partition=partition,
        source_inp_sha256=SHA,
        tfv_m3=tfv,
        pfv_m3=pfv,
        engineering_pass=engineering,
        action_decisions=actions,
        decision_count=10,
        ever_changed_actuator_count=changed,
    )


def passing_panel() -> list[ScientificPanelRow]:
    rows: list[ScientificPanelRow] = []
    for event, scale in (("E1", 1.0), ("E2", 2.0), ("E3", 3.0)):
        rows.extend(
            [
                row(event, "proposed_v30_development", tfv=80.0 * scale, pfv=50.0),
                row(event, "no_control", tfv=100.0 * scale, pfv=60.0, actions=0, changed=0),
                row(event, "matched_internal_rtc", tfv=95.0 * scale, pfv=55.0),
                row(event, "matched_auto_rbc", tfv=90.0 * scale, pfv=52.0),
                row(event, "matched_efd", tfv=92.0 * scale, pfv=54.0, changed=4),
            ]
        )
    return rows


def test_gate_passes_only_when_research_contract_and_active_baselines_pass() -> None:
    result = evaluate_v30_scientific_gate(passing_panel())
    assert result.passed is True
    assert result.event_count == 3
    assert result.issues == ()
    assert result.diagnostics["ready_for_new_policy_lock"] is True


def test_gate_rejects_final_or_locked_partition_even_with_good_numbers() -> None:
    panel = passing_panel()
    panel[0] = row(
        "E1",
        "proposed_v30_development",
        tfv=1.0,
        pfv=1.0,
        partition="final",
    )
    result = evaluate_v30_scientific_gate(panel)
    assert result.passed is False
    assert any("not fresh Development-validation" in issue for issue in result.issues)


def test_gate_rejects_pfv_safety_failure() -> None:
    panel = passing_panel()
    panel[0] = row("E1", "proposed_v30_development", tfv=80.0, pfv=1000.0)
    result = evaluate_v30_scientific_gate(panel)
    assert result.passed is False
    assert any("PFV safety failed" in issue for issue in result.issues)


def test_gate_rejects_tfv_failure_against_strong_baseline() -> None:
    panel = passing_panel()
    for index, item in enumerate(panel):
        if item.strategy == "matched_auto_rbc":
            panel[index] = row(item.event_id, item.strategy, tfv=70.0, pfv=item.pfv_m3)
    result = evaluate_v30_scientific_gate(panel)
    assert result.passed is False
    assert "TFV objective not achieved versus matched_auto_rbc" in result.issues


def test_gate_rejects_degenerate_efd_even_if_tfv_numbers_are_favorable() -> None:
    panel = passing_panel()
    for index, item in enumerate(panel):
        if item.strategy == "matched_efd":
            panel[index] = row(
                item.event_id,
                item.strategy,
                tfv=item.tfv_m3,
                pfv=item.pfv_m3,
                actions=0,
                changed=0,
            )
    result = evaluate_v30_scientific_gate(panel)
    assert result.passed is False
    assert "matched_efd is degenerate/inactive on the Development panel" in result.issues


def test_gate_rejects_incomplete_or_misaligned_event_panel() -> None:
    panel = passing_panel()
    panel = [item for item in panel if not (item.event_id == "E3" and item.strategy == "matched_efd")]
    result = evaluate_v30_scientific_gate(panel)
    assert result.passed is False
    assert any("missing strategies" in issue for issue in result.issues)
