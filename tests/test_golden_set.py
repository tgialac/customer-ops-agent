import json
from collections import Counter
from pathlib import Path

from customer_ops_agent.evaluation import GoldenCase


GOLDEN_SET = Path(__file__).parents[1] / "data" / "golden" / "customer_ops_golden_v1.jsonl"


def load_cases() -> list[GoldenCase]:
    return [
        GoldenCase.model_validate(json.loads(line))
        for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_golden_set_has_balanced_supported_intents() -> None:
    cases = load_cases()
    counts = Counter(case.expected_intent.value for case in cases)

    assert len(cases) == 60
    assert counts == {
        "missing_refund": 20,
        "transaction_pending": 20,
        "transaction_failed": 20,
    }
    assert len({case.case_id for case in cases}) == len(cases)


def test_golden_set_has_actionable_tool_oracle() -> None:
    cases = load_cases()

    assert any(case.expected_tool == "get_refund_status" for case in cases)
    assert any(case.expected_tool == "get_transaction_status" for case in cases)
    assert any(case.expected_tool == "create_support_ticket" for case in cases)
    assert any("multi_turn" in case.tags for case in cases)
    assert any(case.expected_action.value == "ask_clarification" for case in cases)
    assert any(case.expected_action.value == "handoff" for case in cases)
