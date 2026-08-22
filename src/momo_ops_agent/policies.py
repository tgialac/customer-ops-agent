"""Deterministic policy decisions for the first source-backed workflow."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .contracts import FundingSource, StrictModel, TransactionStatus


BANK_TRANSFER_POLICY_SOURCE = "momo-faq-bank-transfer-reversal-2026-08-22"


class BankTransferPolicyAction(str, Enum):
    REQUEST_TRANSACTION_ID = "request_transaction_id"
    EXPLAIN_PENDING_RECONCILIATION = "explain_pending_reconciliation"
    EXPLAIN_BENEFICIARY_POSTING_DELAY = "explain_beneficiary_posting_delay"
    EXPLAIN_FAILED_RETURN = "explain_failed_return"
    HANDOFF = "handoff"


class BankTransferPolicyInput(StrictModel):
    transaction_id: str | None = None
    status: TransactionStatus | None = None
    funding_source: FundingSource = FundingSource.UNKNOWN
    elapsed_working_days: int | None = Field(default=None, ge=0)
    return_elapsed_working_days: int | None = Field(default=None, ge=0)
    wrong_details_reported: bool = False


class BankTransferPolicyDecision(StrictModel):
    action: BankTransferPolicyAction
    source_document_id: str = BANK_TRANSFER_POLICY_SOURCE
    message_key: str
    handoff_required: bool = False
    return_destination: str | None = None


def render_bank_transfer_response(decision: BankTransferPolicyDecision) -> str:
    """Render only the approved source-backed answer variants."""

    messages = {
        "pending_1_to_2_working_days": (
            "Giao dịch đang được đối soát. Vui lòng chờ 1–2 ngày làm việc; "
            "nếu sau thời gian này vẫn chưa có kết quả, bộ phận hỗ trợ sẽ kiểm tra thêm."
        ),
        "successful_transfer_1_to_3_working_days": (
            "Giao dịch đã thành công nhưng ngân hàng người nhận có thể cần "
            "1–3 ngày làm việc để ghi nhận. Nếu quá thời gian này vẫn chưa nhận được tiền, "
            "bộ phận hỗ trợ sẽ kiểm tra thêm."
        ),
        "failed_transfer_return_1_to_2_working_days": (
            "Giao dịch không thành công. Tiền sẽ được hoàn về "
            f"{_return_destination_label(decision.return_destination)} trong khoảng 1–2 "
            "ngày làm việc."
        ),
    }
    try:
        return messages[decision.message_key]
    except KeyError as exc:
        raise ValueError(
            f"no customer-facing response is approved for {decision.message_key}"
        ) from exc


def is_approved_bank_transfer_response(response: str) -> bool:
    """Return whether a response is one of the bounded policy templates."""

    return response in {
        "Giao dịch đang được đối soát. Vui lòng chờ 1–2 ngày làm việc; "
        "nếu sau thời gian này vẫn chưa có kết quả, bộ phận hỗ trợ sẽ kiểm tra thêm.",
        "Giao dịch đã thành công nhưng ngân hàng người nhận có thể cần "
        "1–3 ngày làm việc để ghi nhận. Nếu quá thời gian này vẫn chưa nhận được tiền, "
        "bộ phận hỗ trợ sẽ kiểm tra thêm.",
        "Giao dịch không thành công. Tiền sẽ được hoàn về ví MoMo trong khoảng 1–2 ngày làm việc.",
        "Giao dịch không thành công. Tiền sẽ được hoàn về tài khoản ngân hàng đã liên kết trong khoảng 1–2 ngày làm việc.",
        "Giao dịch không thành công. Tiền sẽ được hoàn về nguồn tiền ban đầu trong khoảng 1–2 ngày làm việc.",
    }


def _return_destination_label(destination: str | None) -> str:
    return {
        "momo_wallet": "ví MoMo",
        "linked_bank": "tài khoản ngân hàng đã liên kết",
        "original_funding_source": "nguồn tiền ban đầu",
    }.get(destination or "", "nguồn tiền ban đầu")


def evaluate_bank_transfer_policy(
    request: BankTransferPolicyInput,
) -> BankTransferPolicyDecision:
    """Map verified transaction facts to a bounded customer-ops outcome.

    The time windows come from the public MoMo FAQ and are deliberately not
    inferred for missing timestamps.  Unknown or out-of-scope facts fail to a
    handoff rather than producing a new promise.
    """

    if request.transaction_id is None:
        return BankTransferPolicyDecision(
            action=BankTransferPolicyAction.REQUEST_TRANSACTION_ID,
            message_key="ask_for_transaction_id",
        )

    if request.wrong_details_reported:
        return BankTransferPolicyDecision(
            action=BankTransferPolicyAction.HANDOFF,
            message_key="wrong_details_need_support_recovery",
            handoff_required=True,
        )

    if request.status is TransactionStatus.PENDING:
        if request.elapsed_working_days is not None and request.elapsed_working_days > 2:
            return _overdue_handoff("pending_transfer_overdue")
        return BankTransferPolicyDecision(
            action=BankTransferPolicyAction.EXPLAIN_PENDING_RECONCILIATION,
            message_key="pending_1_to_2_working_days",
        )

    if request.status is TransactionStatus.COMPLETED:
        if request.elapsed_working_days is not None and request.elapsed_working_days > 3:
            return _overdue_handoff("successful_transfer_overdue")
        return BankTransferPolicyDecision(
            action=BankTransferPolicyAction.EXPLAIN_BENEFICIARY_POSTING_DELAY,
            message_key="successful_transfer_1_to_3_working_days",
        )

    if request.status in {TransactionStatus.FAILED, TransactionStatus.REVERSED}:
        if (
            request.return_elapsed_working_days is not None
            and request.return_elapsed_working_days > 2
        ):
            return _overdue_handoff("failed_transfer_return_overdue")

        destination = {
            FundingSource.WALLET: "momo_wallet",
            FundingSource.LINKED_BANK: "linked_bank",
            FundingSource.UNKNOWN: "original_funding_source",
        }[request.funding_source]
        return BankTransferPolicyDecision(
            action=BankTransferPolicyAction.EXPLAIN_FAILED_RETURN,
            message_key="failed_transfer_return_1_to_2_working_days",
            return_destination=destination,
        )

    return _overdue_handoff("unknown_transfer_state")


def _overdue_handoff(message_key: str) -> BankTransferPolicyDecision:
    return BankTransferPolicyDecision(
        action=BankTransferPolicyAction.HANDOFF,
        message_key=message_key,
        handoff_required=True,
    )
