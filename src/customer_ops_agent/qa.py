"""Human-review artifacts for source-backed answer-generation evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .agent_harness import AgentHarness, AgentRun
from .contracts import StrictModel
from .eval_runner import build_backend, load_cases, load_fixture_config
from .evaluation import GoldenCase
from .knowledge import KnowledgeStore


class AnswerQARecord(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    customer_message: str
    expected_outcome: str
    actual_outcome: str
    expected_status: str
    actual_status: str
    response: str
    response_handle: str
    policy_message_key: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    answer_generation_attempts: int
    checks: dict[str, bool]
    automated_pass: bool
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer_notes: str = ""


class AnswerQAReport(StrictModel):
    schema_version: Literal[1] = 1
    harness: str
    total: int
    automated_passed: int
    records: tuple[AnswerQARecord, ...]


_REVIEW_FINGERPRINT_FIELDS = (
    "customer_message",
    "response",
    "actual_outcome",
    "policy_message_key",
    "source_document_id",
)


class ReviewGateReport(StrictModel):
    """Release decision after automated checks and human review are combined."""

    schema_version: Literal[1] = 1
    passed: bool
    total: int
    automated_passed: int
    approved: int
    pending: int
    rejected: int
    automated_failures: tuple[str, ...]
    pending_cases: tuple[str, ...]
    rejected_cases: tuple[str, ...]


def evaluate_review_gate(report: AnswerQAReport) -> ReviewGateReport:
    """Require every case to pass automation and receive explicit approval."""

    automated_failures = tuple(
        record.case_id for record in report.records if not record.automated_pass
    )
    approved = tuple(record.case_id for record in report.records if record.review_status == "approved")
    pending = tuple(record.case_id for record in report.records if record.review_status == "pending")
    rejected = tuple(record.case_id for record in report.records if record.review_status == "rejected")
    return ReviewGateReport(
        passed=bool(report.records)
        and not automated_failures
        and not pending
        and not rejected
        and len(approved) == len(report.records),
        total=report.total,
        automated_passed=report.automated_passed,
        approved=len(approved),
        pending=len(pending),
        rejected=len(rejected),
        automated_failures=automated_failures,
        pending_cases=pending,
        rejected_cases=rejected,
    )


def _grade_case(case: GoldenCase, run: AgentRun) -> AnswerQARecord:
    # Keep the artifact independent from the broader EvalRecord so reviewers
    # see the actual customer-facing answer and its grounding metadata.
    agent_run = run
    actual = agent_run.final_decision
    trace = agent_run.trace[-1]
    expected_lookup = (
        case.expected_lookup_tool.value if case.expected_lookup_tool else None
    )
    actual_lookup = trace.tool_result.tool_name.value if trace.tool_result else None
    source_url = None
    if actual.policy_source:
        source_url = _QA_KNOWLEDGE.get(actual.policy_source)
        source_url = source_url.source_url if source_url else None
    checks = {
        "intent": actual.intent.intent is case.expected_intent,
        "slots": actual.slots == case.expected_slots,
        "action": actual.action is case.expected_action,
        "status": agent_run.case_state.status is case.expected_status,
        "outcome": actual.outcome == case.expected_outcome,
        "lookup_tool": expected_lookup is None or actual_lookup == expected_lookup,
        "output_guardrail": (
            trace.output_guardrail is not None and trace.output_guardrail.passed
        ),
    }
    return AnswerQARecord(
        case_id=case.case_id,
        customer_message=case.turns[-1].text,
        expected_outcome=case.expected_outcome.value,
        actual_outcome=actual.outcome.value,
        expected_status=case.expected_status.value,
        actual_status=agent_run.case_state.status.value,
        response=actual.customer_response or actual.response,
        response_handle=(
            actual.action.value if actual.action.value == "answer" else actual.response
        ),
        policy_message_key=actual.policy_message_key,
        source_document_id=actual.policy_source,
        source_url=source_url,
        answer_generation_attempts=trace.answer_generation_attempts,
        checks=checks,
        automated_pass=all(checks.values()),
    )


_QA_KNOWLEDGE = KnowledgeStore.from_directory(
    Path(__file__).parents[2] / "data" / "knowledge" / "policies"
)


def run_answer_qa(
    golden_path: Path,
    fixture_path: Path,
    agent: AgentHarness,
    *,
    previous_path: Path | None = None,
) -> AnswerQAReport:
    cases = load_cases(golden_path)
    fixtures = load_fixture_config(fixture_path)
    previous: dict[str, dict[str, str]] = {}
    if previous_path is not None and previous_path.exists():
        payload = json.loads(previous_path.read_text(encoding="utf-8"))
        previous = {
            record["case_id"]: record
            for record in payload.get("records", [])
        }

    records: list[AnswerQARecord] = []
    for case in cases:
        run = agent.run(
            case.case_id,
            case.turns,
            build_backend(case, fixtures),
        )
        record = _grade_case(case, run)
        old = previous.get(case.case_id)
        review_target_unchanged = old is not None and all(
            old.get(field) == getattr(record, field)
            for field in _REVIEW_FINGERPRINT_FIELDS
        )
        if review_target_unchanged:
            record = record.model_copy(
                update={
                    "review_status": old.get("review_status", "pending"),
                    "reviewer_notes": old.get("reviewer_notes", ""),
                }
            )
        records.append(record)

    return AnswerQAReport(
        harness=agent.harness_name,
        total=len(records),
        automated_passed=sum(record.automated_pass for record in records),
        records=tuple(records),
    )
