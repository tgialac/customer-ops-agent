from pathlib import Path

from momo_ops_agent.agent_harness import AgentDecision, LLMAgent, RuleBasedAgent
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
    def provider(history: list[GoldenTurn]) -> AgentDecision:
        assert history[-1].role is GoldenTurnRole.CUSTOMER
        return AgentDecision(
            intent=IntentPrediction(
                intent=IntentName.TRANSACTION_PENDING,
                confidence=0.96,
                source="classifier",
            ),
            slots={SlotName.TRANSACTION_ID: "txn_demo_101"},
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=ToolName.GET_TRANSACTION_STATUS,
            tool_args={"transaction_id": "txn_demo_101"},
            outcome="retrieve_transaction_status",
            response="Mình đang kiểm tra giao dịch.",
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


def test_semantic_contract_violation_becomes_safe_handoff() -> None:
    def provider(_: list[GoldenTurn]) -> AgentDecision:
        return AgentDecision(
            intent=IntentPrediction(
                intent=IntentName.TRANSACTION_PENDING,
                confidence=0.99,
                source="classifier",
            ),
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=ToolName.GET_TRANSACTION_STATUS,
            tool_args={},
            outcome="retrieve_transaction_status",
            response="Mình kiểm tra nhé.",
        )

    backend = MockBackend()
    run = LLMAgent(provider).run(
        "llm-test-102",
        [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text="Giao dịch đang chờ xử lý")],
        backend,
    )

    assert run.final_decision.outcome == "policy_guardrail_handoff"
    assert run.case_state.status is CaseStatus.HANDED_OFF
    assert run.trace[0].tool_result is None
    assert backend.snapshot()["audit_log"] == []


def test_injected_decision_provider_uses_the_same_evaluator() -> None:
    baseline = RuleBasedAgent()
    summary = run_evaluation(
        ROOT / "data/golden/momo_golden_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        LLMAgent(baseline.decide),
    )

    assert summary.harness == "llm_decision_v1"
    assert summary.passed == 60
