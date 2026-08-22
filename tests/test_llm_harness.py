from pathlib import Path

from momo_ops_agent.agent_harness import (
    LLMDecisionPayload,
    LLMAgent,
    RouterDecision,
    RuleBasedAgent,
)
from momo_ops_agent.contracts import (
    CaseAction,
    CaseStatus,
    IntentName,
    IntentPrediction,
    SlotName,
    TransactionStatus,
)
from momo_ops_agent.evaluation import GoldenTurn, GoldenTurnRole, ToolName
from momo_ops_agent.mock_backend import MockBackend, TransactionRecord
from momo_ops_agent.eval_runner import run_evaluation


ROOT = Path(__file__).parents[1]


def test_injected_llm_decision_runs_through_stateful_backend() -> None:
    def provider(history: list[GoldenTurn]) -> RouterDecision:
        assert history[-1].role is GoldenTurnRole.CUSTOMER
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.TRANSACTION_PENDING,
                confidence=0.96,
                source="classifier",
            ),
            slots={SlotName.TRANSACTION_ID: "txn_demo_101"},
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=ToolName.GET_TRANSACTION_STATUS,
            tool_args={"transaction_id": "txn_demo_101"},
        )

    backend = MockBackend(
        [TransactionRecord(transaction_id="txn_demo_101", status=TransactionStatus.PENDING)]
    )
    run = LLMAgent(provider).run(
        "llm-test-101",
        [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text="txn_demo_101 đang pending")],
        backend,
    )

    assert run.final_decision.action is CaseAction.RETRIEVE_CONTEXT
    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.data["status"] == "pending"
    assert run.case_state.status is CaseStatus.IN_PROGRESS


def test_openai_wire_schema_avoids_dynamic_property_names() -> None:
    schema = LLMDecisionPayload.model_json_schema()

    assert "propertyNames" not in str(schema)
    payload = LLMDecisionPayload(
        intent=IntentName.TRANSACTION_PENDING,
        confidence=0.96,
        transaction_id="txn_demo_101",
        action=CaseAction.RETRIEVE_CONTEXT,
        tool=ToolName.GET_TRANSACTION_STATUS,
    )
    decision = payload.to_router_decision()

    assert decision.slots[SlotName.TRANSACTION_ID] == "txn_demo_101"
    assert decision.tool_args["transaction_id"] == "txn_demo_101"


def test_missing_tool_slot_is_safely_normalized_to_clarification() -> None:
    def provider(_: list[GoldenTurn]) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.TRANSACTION_PENDING,
                confidence=0.99,
                source="classifier",
            ),
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=ToolName.GET_TRANSACTION_STATUS,
            tool_args={},
        )

    backend = MockBackend()
    run = LLMAgent(provider).run(
        "llm-test-102",
        [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text="Giao dịch đang chờ xử lý")],
        backend,
    )

    assert run.final_decision.action is CaseAction.ASK_CLARIFICATION
    assert run.final_decision.outcome == "ask_for_transaction_id"
    assert run.case_state.status is CaseStatus.WAITING_FOR_CUSTOMER
    assert run.trace[0].tool_result is None
    assert backend.snapshot()["audit_log"] == []


def test_router_normalization_recovers_explicit_id_and_ticket_policy() -> None:
    def provider(_: list[GoldenTurn]) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.MISSING_REFUND,
                confidence=0.99,
                source="classifier",
            ),
            action=CaseAction.ASK_CLARIFICATION,
        )

    backend = MockBackend(
        [TransactionRecord(transaction_id="txn_demo_009", status=TransactionStatus.COMPLETED)]
    )
    run = LLMAgent(provider).run(
        "llm-test-103",
        [
            GoldenTurn(
                role=GoldenTurnRole.CUSTOMER,
                text="Refund của txn_demo_009 bị lỗi rồi, mình cần được hỗ trợ.",
            )
        ],
        backend,
    )

    assert run.final_decision.action is CaseAction.EXECUTE_TOOL
    assert run.final_decision.tool is ToolName.CREATE_SUPPORT_TICKET
    assert run.trace[0].tool_result is not None
    assert run.case_state.status is CaseStatus.IN_PROGRESS


def test_router_normalization_canonicalizes_explicit_intent_and_id() -> None:
    def provider(_: list[GoldenTurn]) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.TRANSACTION_PENDING,
                confidence=0.99,
                source="classifier",
            ),
            slots={SlotName.TRANSACTION_ID: "txn_demo_999"},
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=ToolName.GET_TRANSACTION_STATUS,
            tool_args={"transaction_id": "txn_demo_999"},
        )

    backend = MockBackend(
        [TransactionRecord(transaction_id="txn_demo_042", status=TransactionStatus.FAILED)]
    )
    run = LLMAgent(provider).run(
        "llm-test-104",
        [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text="Txn_demo_042 bị failed")],
        backend,
    )

    assert run.final_decision.intent.intent is IntentName.TRANSACTION_FAILED
    assert run.final_decision.slots[SlotName.TRANSACTION_ID] == "txn_demo_042"
    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.success is True


def test_source_workflow_cannot_answer_before_transaction_lookup() -> None:
    def provider(_: list[GoldenTurn]) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.BANK_TRANSFER_NOT_RECEIVED,
                confidence=0.99,
                source="classifier",
            ),
            action=CaseAction.ANSWER,
        )

    transaction_id = "txn_demo_072"
    run = LLMAgent(provider).run(
        "llm-bank-transfer-001",
        [
            GoldenTurn(
                role=GoldenTurnRole.CUSTOMER,
                text=f"Chuyển khoản ngân hàng, người nhận chưa thấy tiền, mã {transaction_id}.",
            )
        ],
        MockBackend(
            [
                TransactionRecord(
                    transaction_id=transaction_id,
                    status=TransactionStatus.PENDING,
                    elapsed_working_days=1,
                )
            ]
        ),
    )

    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.outcome == "bank_transfer_pending_reconciliation"
    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.tool_name is ToolName.GET_TRANSACTION_STATUS


def test_injected_decision_provider_uses_the_same_evaluator() -> None:
    baseline = RuleBasedAgent()
    summary = run_evaluation(
        ROOT / "data/golden/momo_golden_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        LLMAgent(baseline.decide),
    )

    assert summary.harness == "llm_decision_v1"
    assert summary.passed == 60
