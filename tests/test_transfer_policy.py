from pathlib import Path

from momo_ops_agent.contracts import TransactionStatus
from momo_ops_agent.knowledge import KnowledgeStore
from momo_ops_agent.policies import (
    BANK_TRANSFER_POLICY_SOURCE,
    BankTransferPolicyAction,
    BankTransferPolicyInput,
    FundingSource,
    evaluate_bank_transfer_policy,
    render_bank_transfer_response,
)


ROOT = Path(__file__).parents[1]


def test_missing_transaction_id_is_clarification_not_a_policy_claim() -> None:
    decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(status=TransactionStatus.PENDING)
    )

    assert decision.action is BankTransferPolicyAction.REQUEST_TRANSACTION_ID
    assert decision.handoff_required is False


def test_policy_rule_points_to_an_active_source_document() -> None:
    store = KnowledgeStore.from_directory(ROOT / "data/knowledge/momo")

    documents = [
        hit.document
        for hit in store.search(
            "giao dịch ngân hàng pending",
            topic="bank_transfer_reversal",
        )
    ]

    assert any(
        document.document_id == BANK_TRANSFER_POLICY_SOURCE
        and document.status == "active"
        for document in documents
    )


def test_pending_transfer_uses_momo_reconciliation_window() -> None:
    within_window = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_201",
            status=TransactionStatus.PENDING,
            elapsed_working_days=2,
        )
    )
    overdue = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_202",
            status=TransactionStatus.PENDING,
            elapsed_working_days=3,
        )
    )

    assert within_window.message_key == "pending_1_to_2_working_days"
    assert within_window.handoff_required is False
    assert overdue.action is BankTransferPolicyAction.HANDOFF


def test_successful_transfer_uses_beneficiary_posting_window() -> None:
    decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_203",
            status=TransactionStatus.COMPLETED,
            elapsed_working_days=3,
        )
    )

    assert decision.action is BankTransferPolicyAction.EXPLAIN_BENEFICIARY_POSTING_DELAY
    assert decision.message_key == "successful_transfer_1_to_3_working_days"


def test_failed_transfer_explains_return_destination_without_promising_refund() -> None:
    wallet_decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_204",
            status=TransactionStatus.FAILED,
            funding_source=FundingSource.WALLET,
        )
    )
    bank_decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_205",
            status=TransactionStatus.FAILED,
            funding_source=FundingSource.LINKED_BANK,
            return_elapsed_working_days=2,
        )
    )

    assert wallet_decision.action is BankTransferPolicyAction.EXPLAIN_FAILED_RETURN
    assert wallet_decision.return_destination == "momo_wallet"
    assert bank_decision.return_destination == "linked_bank"


def test_failed_transfer_return_overdue_is_handoff() -> None:
    decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_206",
            status=TransactionStatus.FAILED,
            return_elapsed_working_days=3,
        )
    )

    assert decision.action is BankTransferPolicyAction.HANDOFF
    assert decision.handoff_required is True


def test_wrong_details_are_support_recovery_not_automatic_refund() -> None:
    decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_207",
            status=TransactionStatus.COMPLETED,
            wrong_details_reported=True,
        )
    )

    assert decision.action is BankTransferPolicyAction.HANDOFF
    assert decision.message_key == "wrong_details_need_support_recovery"


def test_source_backed_response_renderer_has_bounded_wording() -> None:
    decision = evaluate_bank_transfer_policy(
        BankTransferPolicyInput(
            transaction_id="txn_demo_208",
            status=TransactionStatus.FAILED,
            funding_source=FundingSource.LINKED_BANK,
            return_elapsed_working_days=1,
        )
    )

    response = render_bank_transfer_response(decision)

    assert "1–2 ngày làm việc" in response
    assert "tài khoản ngân hàng đã liên kết" in response
    assert "hoàn" in response
