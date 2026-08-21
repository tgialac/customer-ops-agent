from pathlib import Path

from momo_ops_agent.eval_runner import run_evaluation


ROOT = Path(__file__).parents[1]


def test_rule_based_baseline_runs_every_golden_case() -> None:
    summary = run_evaluation(
        ROOT / "data/golden/momo_golden_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
    )

    assert summary.harness == "rule_based_v1"
    assert summary.total == 60
    assert summary.passed == 60
    assert summary.pass_rate == 1.0
