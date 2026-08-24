from pathlib import Path

from customer_ops_agent.agent_harness import RuleBasedAgent
from customer_ops_agent.answering import DeterministicAnswerGenerator, KnowledgeBackedAnswerer
from customer_ops_agent.contracts import CaseAction, CaseStatus, IntentName, RefundStatus
from customer_ops_agent.evaluation import GoldenTurn, GoldenTurnRole, OutcomeName, ToolName
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord
from customer_ops_agent.policies import (
    CashbackPolicyInput,
    evaluate_cashback_policy,
)


ROOT = Path(__file__).parents[1]


def _turn(text: str) -> list[GoldenTurn]:
    return [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text=text)]


def _backend(transaction_id: str, **updates: object) -> MockBackend:
    values = {
        "transaction_id": transaction_id,
        "status": "completed",
        "refund_id": f"refund_for_{transaction_id}",
        "refund_status": RefundStatus.PROCESSING,
        "cashback_elapsed_hours": 12,
    }
    values.update(updates)
    return MockBackend([TransactionRecord(**values)])


def test_cashback_policy_hands_off_after_public_window() -> None:
    decision = evaluate_cashback_policy(
        CashbackPolicyInput(transaction_id="txn_demo_200", elapsed_hours=25)
    )

    assert decision.handoff_required is True
    assert decision.message_key == "cashback_overdue_help"


def test_cashback_policy_hands_off_when_elapsed_time_is_unavailable() -> None:
    decision = evaluate_cashback_policy(
        CashbackPolicyInput(transaction_id="txn_demo_200")
    )

    assert decision.handoff_required is True
    assert decision.message_key == "cashback_elapsed_time_unknown"


def test_cashback_missing_id_asks_for_cashback_transaction_id() -> None:
    run = RuleBasedAgent().run(
        "cashback-runtime-000",
        _turn("Tôi chưa nhận được khoản hoàn tiền cashback."),
        MockBackend(),
    )

    assert run.final_decision.action is CaseAction.ASK_CLARIFICATION
    assert "cashback" in (run.final_decision.customer_response or "")


def test_cashback_workflow_materializes_verified_policy() -> None:
    run = RuleBasedAgent().run(
        "cashback-runtime-001",
        _turn("Khoản hoàn cashback của txn_demo_201 chưa về."),
        _backend("txn_demo_201"),
    )

    assert run.final_decision.intent.intent is IntentName.MISSING_REFUND
    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.outcome is OutcomeName.CASHBACK_PENDING_WITHIN_24_HOURS
    assert run.final_decision.policy_source == (
        "official-faq-cashback-not-received-2026-08-22"
    )
    assert run.final_decision.policy_message_key == "cashback_pending_within_24_hours"
    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.tool_name is ToolName.GET_REFUND_STATUS
    assert run.case_state.context.refund is not None
    assert run.case_state.status is CaseStatus.IN_PROGRESS


def test_cashback_answer_layer_retrieves_cashback_source() -> None:
    answerer = KnowledgeBackedAnswerer.from_repository(DeterministicAnswerGenerator())
    run = RuleBasedAgent(answerer=answerer).run(
        "cashback-runtime-002",
        _turn("Khoản hoàn cashback của txn_demo_202 chưa về."),
        _backend("txn_demo_202"),
    )

    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.customer_response is not None
    assert "24 giờ" in run.final_decision.customer_response
    assert run.trace[0].answer_generation_attempts == 1
    assert run.trace[0].output_guardrail is not None
    assert run.trace[0].output_guardrail.passed is True


def test_cashback_overdue_case_handoffs_without_answer_generation() -> None:
    run = RuleBasedAgent().run(
        "cashback-runtime-003",
        _turn("Cashback của txn_demo_203 chưa về dù đã hơn một ngày."),
        _backend("txn_demo_203", cashback_elapsed_hours=25),
    )

    assert run.final_decision.action is CaseAction.HANDOFF
    assert run.final_decision.outcome is OutcomeName.CASHBACK_HANDOFF
    assert run.final_decision.response == "handoff"
    assert run.final_decision.customer_response


def test_cashback_missing_elapsed_time_handoffs_after_lookup() -> None:
    run = RuleBasedAgent().run(
        "cashback-missing-time-001",
        _turn("Cashback của txn_demo_204 chưa về."),
        _backend("txn_demo_204", cashback_elapsed_hours=None),
    )

    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.tool_name is ToolName.GET_REFUND_STATUS
    assert run.case_state.context.refund is not None
    assert run.case_state.context.refund.cashback_elapsed_hours is None
    assert run.final_decision.action is CaseAction.HANDOFF
    assert run.final_decision.outcome is OutcomeName.CASHBACK_HANDOFF
    assert run.final_decision.response == "handoff"
    assert run.case_state.status is CaseStatus.HANDED_OFF
