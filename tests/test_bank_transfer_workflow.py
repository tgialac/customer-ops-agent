from pathlib import Path

from momo_ops_agent.agent_harness import RuleBasedAgent
from momo_ops_agent.contracts import CaseAction, CaseStatus, IntentName, TransactionStatus
from momo_ops_agent.eval_runner import run_evaluation
from momo_ops_agent.evaluation import GoldenTurn, GoldenTurnRole, OutcomeName, ToolName
from momo_ops_agent.mock_backend import MockBackend, TransactionRecord


ROOT = Path(__file__).parents[1]


def test_source_backed_bank_transfer_suite_passes() -> None:
    summary = run_evaluation(
        ROOT / "data/golden/bank_transfer_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
    )

    assert summary.total == 11
    assert summary.passed == 11
    assert summary.pass_rate == 1.0


def test_bank_transfer_lookup_is_materialized_from_verified_context() -> None:
    transaction_id = "txn_demo_070"
    run = RuleBasedAgent().run(
        "bank-transfer-context-001",
        [
            GoldenTurn(
                role=GoldenTurnRole.CUSTOMER,
                text=(
                    "Chuyển khoản ngân hàng, người nhận chưa thấy tiền, "
                    f"mã {transaction_id}."
                ),
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

    assert run.final_decision.intent.intent is IntentName.BANK_TRANSFER_NOT_RECEIVED
    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.outcome is OutcomeName.BANK_TRANSFER_PENDING_RECONCILIATION
    assert run.final_decision.policy_source == (
        "momo-faq-bank-transfer-reversal-2026-08-22"
    )
    assert run.trace[0].tool_result is not None
    assert run.trace[0].tool_result.tool_name is ToolName.GET_TRANSACTION_STATUS
    assert run.case_state.context.transaction is not None
    assert run.case_state.context.transaction.elapsed_working_days == 1
    assert run.case_state.status is CaseStatus.IN_PROGRESS


def test_bank_transfer_lookup_failure_handoffs_without_answering() -> None:
    run = RuleBasedAgent().run(
        "bank-transfer-failure-001",
        [
            GoldenTurn(
                role=GoldenTurnRole.CUSTOMER,
                text="Chuyển khoản ngân hàng, người nhận chưa thấy tiền, mã txn_demo_071.",
            )
        ],
        MockBackend(),
    )

    assert run.final_decision.action is CaseAction.HANDOFF
    assert run.final_decision.outcome is OutcomeName.BANK_TRANSFER_TOOL_FAILURE_HANDOFF
    assert run.case_state.status is CaseStatus.HANDED_OFF
    assert run.final_decision.response == "handoff"
