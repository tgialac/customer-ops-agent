"""Run the golden set against an agent harness and grade the outcome."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import Field

from .agent_harness import AgentHarness, AgentRun, LLMAgent, RuleBasedAgent
from .contracts import RefundStatus, StrictModel, TransactionStatus
from .evaluation import GoldenCase
from .mock_backend import MockBackend, TransactionRecord


class EvalRecord(StrictModel):
    case_id: str
    expected_intent: str
    actual_intent: str
    expected_action: str
    actual_action: str
    expected_tool: str | None
    actual_tool: str | None
    expected_lookup_tool: str | None
    actual_lookup_tool: str | None
    expected_status: str
    actual_status: str
    expected_outcome: str
    actual_outcome: str
    expected_policy_source: str | None
    actual_policy_source: str | None
    expected_policy_message_key: str | None
    actual_policy_message_key: str | None
    checks: dict[str, bool]
    passed: bool


class EvalSummary(StrictModel):
    harness: str
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    by_intent: dict[str, dict[str, int]]
    records: tuple[EvalRecord, ...]


def load_cases(path: Path) -> list[GoldenCase]:
    return [
        GoldenCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fixture_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_backend(case: GoldenCase, fixture_config: dict[str, Any]) -> MockBackend:
    transaction_id = case.expected_slots.get("transaction_id")
    if transaction_id is None:
        return MockBackend()

    defaults = fixture_config["default_by_intent"][case.expected_intent.value]
    override = fixture_config.get("overrides", {}).get(case.case_id, {})
    state = {**defaults, **override}
    refund_status = state.get("refund_status")
    refund_id = case.expected_slots.get("refund_id")
    record = TransactionRecord(
        transaction_id=transaction_id,
        status=TransactionStatus(state["status"]),
        refund_id=refund_id or f"refund_for_{transaction_id}",
        refund_status=RefundStatus(refund_status) if refund_status else None,
        funding_source=state.get("funding_source", "unknown"),
        elapsed_working_days=state.get("elapsed_working_days"),
        return_elapsed_working_days=state.get("return_elapsed_working_days"),
        cashback_elapsed_hours=state.get("cashback_elapsed_hours"),
        cashback_reason=state.get("cashback_reason"),
    )
    return MockBackend([record])


def evaluate_case(
    case: GoldenCase,
    fixture_config: dict[str, Any],
    agent: AgentHarness | None = None,
) -> EvalRecord:
    harness = agent or RuleBasedAgent()
    run: AgentRun = harness.run(case.case_id, case.turns, build_backend(case, fixture_config))
    actual = run.final_decision
    expected_tool = case.expected_tool.value if case.expected_tool else None
    actual_tool = actual.tool.value if actual.tool else None
    expected_lookup_tool = (
        case.expected_lookup_tool.value if case.expected_lookup_tool else None
    )
    lookup_result = run.trace[-1].tool_result
    actual_lookup_tool = lookup_result.tool_name.value if lookup_result else None
    checks = {
        "intent": actual.intent.intent is case.expected_intent,
        "slots": actual.slots == case.expected_slots,
        "action": actual.action is case.expected_action,
        "tool": actual_tool == expected_tool,
        "lookup_tool": (
            expected_lookup_tool is None or actual_lookup_tool == expected_lookup_tool
        ),
        "status": run.case_state.status is case.expected_status,
        "outcome": actual.outcome == case.expected_outcome,
        "policy_source": (
            case.expected_policy_source is None
            or actual.policy_source == case.expected_policy_source
        ),
        "policy_message_key": (
            case.expected_policy_message_key is None
            or actual.policy_message_key == case.expected_policy_message_key
        ),
    }
    return EvalRecord(
        case_id=case.case_id,
        expected_intent=case.expected_intent.value,
        actual_intent=actual.intent.intent.value,
        expected_action=case.expected_action.value,
        actual_action=actual.action.value,
        expected_tool=expected_tool,
        actual_tool=actual_tool,
        expected_lookup_tool=expected_lookup_tool,
        actual_lookup_tool=actual_lookup_tool,
        expected_status=case.expected_status.value,
        actual_status=run.case_state.status.value,
        expected_outcome=case.expected_outcome,
        actual_outcome=actual.outcome,
        expected_policy_source=case.expected_policy_source,
        actual_policy_source=actual.policy_source,
        expected_policy_message_key=case.expected_policy_message_key,
        actual_policy_message_key=actual.policy_message_key,
        checks=checks,
        passed=all(checks.values()),
    )


def run_evaluation(
    golden_path: Path,
    fixture_path: Path,
    agent: AgentHarness | None = None,
) -> EvalSummary:
    cases = load_cases(golden_path)
    fixtures = load_fixture_config(fixture_path)
    harness = agent or RuleBasedAgent()
    records = tuple(evaluate_case(case, fixtures, harness) for case in cases)
    by_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for record in records:
        counts = by_intent[record.expected_intent]
        counts["total"] += 1
        counts["passed"] += int(record.passed)
    passed = sum(record.passed for record in records)
    return EvalSummary(
        harness=harness.harness_name,
        total=len(records),
        passed=passed,
        pass_rate=passed / len(records) if records else 0.0,
        by_intent=dict(by_intent),
        records=records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=Path("data/golden/customer_ops_golden_v1.jsonl"),
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/golden/fixtures.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--harness",
        choices=("rule", "openai"),
        default="rule",
        help="offline rule baseline or optional OpenAI Responses adapter",
    )
    parser.add_argument("--model", help="model for --harness openai")
    args = parser.parse_args()
    agent: AgentHarness = (
        LLMAgent.from_environment(args.model)
        if args.harness == "openai"
        else RuleBasedAgent()
    )
    summary = run_evaluation(args.golden_set, args.fixtures, agent)
    payload = summary.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("harness", "total", "passed", "pass_rate", "by_intent")}, indent=2))


if __name__ == "__main__":
    main()
