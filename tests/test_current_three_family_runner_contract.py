from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_RUNNER = ROOT / "scripts" / "run_policy_direct_tfv_base_hybrid_parent_current.py"
POLICY_RETURN_RUNNER = ROOT / "scripts" / "run_policy_direct_tfv_policy_return_development.py"
BASE_FACTORY = ROOT / "src" / "rtc" / "direct_tfv_base_probe_runtime_factory.py"
POLICY_RETURN_FACTORY = ROOT / "src" / "rtc" / "direct_tfv_policy_return_runtime_factory.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_three_family_runner_contract(text: str) -> None:
    assert 'lineage.get("projected_gradient_h10_enabled") is not False' in text
    assert 'lineage.get("projected_gradient_cli_knobs_affect_current_policy") is not False' in text
    assert 'lineage.get("candidate_portfolio_family_count_max", -1)) != 3' in text
    assert '"projected_gradient_h10_enabled": False' in text
    assert '"projected_gradient_role": "DEVELOPMENT_ABLATION_ONLY"' in text
    assert '"projected_gradient_cli_knobs_affect_current_policy": False' in text
    assert '"candidate_portfolio_family_count_max": 3' in text
    assert '"candidate_portfolio_family_count_max": 4' not in text
    assert '"projected_gradient_free_dimension"' not in text


def test_base_parent_runner_matches_current_three_family_factory_lineage() -> None:
    runner = _text(PARENT_RUNNER)
    factory = _text(BASE_FACTORY)
    _assert_three_family_runner_contract(runner)
    assert '"projected_gradient_h10_enabled": False' in factory
    assert '"projected_gradient_cli_knobs_affect_current_policy": False' in factory
    assert '"candidate_portfolio_family_count_max": 3' in factory


def test_policy_return_runner_matches_current_three_family_factory_lineage() -> None:
    runner = _text(POLICY_RETURN_RUNNER)
    factory = _text(POLICY_RETURN_FACTORY)
    _assert_three_family_runner_contract(runner)
    assert '"projected_gradient_h10_enabled": False' in factory
    assert '"projected_gradient_cli_knobs_affect_current_policy": False' in factory
    assert '"candidate_portfolio_family_count_max": 3' in factory


def test_current_runners_keep_gradient_cli_knobs_compatibility_only() -> None:
    for path in (PARENT_RUNNER, POLICY_RETURN_RUNNER):
        text = _text(path)
        assert 'p.add_argument("--projected-gradient-steps"' in text
        assert 'p.add_argument("--projected-gradient-step-fraction"' in text
        assert "compatib" in text.lower()
        assert "unexpectedly enabled projected" in text.lower()
