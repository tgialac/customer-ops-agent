"""Deterministic policy decisions for the first source-backed workflow."""

from __future__ import annotations

from enum import Enum
import re

from pydantic import Field

from .contracts import FundingSource, StrictModel, TransactionStatus


BANK_TRANSFER_POLICY_SOURCE = "momo-faq-bank-transfer-reversal-2026-08-22"
CASHBACK_POLICY_SOURCE = "momo-faq-cashback-not-received-2026-08-22"
GOOGLE_PLAY_REFUND_POLICY_SOURCE = "momo-faq-google-play-refund-2026-08-22"


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


class CashbackPolicyAction(str, Enum):
    WAIT_WITHIN_24_HOURS = "wait_within_24_hours"
    EXPLAIN_NOT_ELIGIBLE = "explain_not_eligible"
    EXPLAIN_ACCOUNT_LIMIT = "explain_account_limit"
    EXPLAIN_MONTHLY_LIMIT = "explain_monthly_limit"
    HANDOFF = "handoff"


class CashbackPolicyInput(StrictModel):
    transaction_id: str | None = None
    elapsed_hours: int | None = Field(default=None, ge=0)
    reason: str | None = None


class CashbackPolicyDecision(StrictModel):
    action: CashbackPolicyAction
    source_document_id: str = CASHBACK_POLICY_SOURCE
    message_key: str
    handoff_required: bool = False


class GooglePlayRefundPolicyDecision(StrictModel):
    source_document_id: str = GOOGLE_PLAY_REFUND_POLICY_SOURCE
    message_key: str = "google_play_refund_request_steps"


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


def is_grounded_bank_transfer_response(
    response: str,
    *,
    message_key: str,
    return_destination: str | None = None,
) -> bool:
    """Check mandatory policy facts without requiring an exact sentence match."""

    lowered = response.casefold()
    if len(response) > 2_000 or any(
        phrase in lowered
        for phrase in ("chắc chắn", "ngay lập tức", "trong vài phút", "đảm bảo")
    ):
        return False

    requirements = {
        "pending_1_to_2_working_days": (
            r"1\s*[–-]\s*2",
            "ngày làm việc",
            ("đối soát", "hỗ trợ"),
        ),
        "successful_transfer_1_to_3_working_days": (
            r"1\s*[–-]\s*3",
            "ngày làm việc",
            ("ngân hàng", "hỗ trợ"),
        ),
        "failed_transfer_return_1_to_2_working_days": (
            r"1\s*[–-]\s*2",
            "ngày làm việc",
            ("hoàn",),
        ),
    }
    required = requirements.get(message_key)
    if required is None:
        return False
    if not re.search(required[0], lowered) or required[1] not in lowered:
        return False
    if not any(term in lowered for term in required[2]):
        return False
    if message_key == "failed_transfer_return_1_to_2_working_days":
        destination_labels = {
            "momo_wallet": "ví momo",
            "linked_bank": "tài khoản ngân hàng đã liên kết",
            "original_funding_source": "nguồn tiền ban đầu",
        }
        expected = destination_labels.get(return_destination or "")
        if expected is not None and expected not in lowered:
            return False
    return True


def render_cashback_response(decision: CashbackPolicyDecision) -> str:
    messages = {
        "cashback_pending_within_24_hours": (
            "Hệ thống có thể cần tối đa 24 giờ để ghi nhận tiền hoàn. "
            "Bạn vui lòng chờ thêm và kiểm tra lại trong mục Lịch sử giao dịch."
        ),
        "cashback_not_eligible": (
            "Giao dịch này không thuộc nhóm dịch vụ được áp dụng hoàn tiền theo chính sách. "
            "Bạn có thể kiểm tra danh sách dịch vụ áp dụng trong tính năng Hoàn tiền trên MoMo."
        ),
        "cashback_account_limit_reached": (
            "Tài khoản Hoàn tiền đã đạt giới hạn 12.000.000 đồng. "
            "Bạn cần rút tiền về Ví MoMo để có thể tiếp tục nhận hoàn tiền cho các dịch vụ được áp dụng."
        ),
        "cashback_monthly_limit_reached": (
            "Bạn đã đạt giới hạn hoàn tiền 2.000.000 đồng trong tháng này. "
            "Bạn có thể tiếp tục sử dụng dịch vụ vào tháng sau để nhận hoàn tiền."
        ),
    }
    try:
        return messages[decision.message_key]
    except KeyError as exc:
        raise ValueError(
            f"no customer-facing response is approved for {decision.message_key}"
        ) from exc


def is_grounded_cashback_response(response: str, *, message_key: str) -> bool:
    lowered = response.casefold()
    if len(response) > 2_000 or any(
        phrase in lowered
        for phrase in ("chắc chắn", "ngay lập tức", "đảm bảo", "cam kết")
    ):
        return False
    requirements = {
        "cashback_pending_within_24_hours": ("24 giờ", ("chờ", "lịch sử")),
        "cashback_not_eligible": ("không thuộc", ("dịch vụ", "hoàn tiền")),
        "cashback_account_limit_reached": (
            "12.000.000",
            ("tài khoản hoàn tiền", "rút"),
        ),
        "cashback_monthly_limit_reached": (
            "2.000.000",
            ("tháng", "hoàn tiền"),
        ),
    }
    required = requirements.get(message_key)
    if required is None:
        return False
    return required[0] in lowered and all(term in lowered for term in required[1])


def render_google_play_refund_response(
    message_key: str = "google_play_refund_request_steps",
) -> str:
    messages = {
        "google_play_refund_request_steps": (
            "Để yêu cầu hoàn tiền cho ứng dụng đã mua, bạn mở Google Play, chọn "
            "Tài khoản > Lịch sử đơn đặt hàng, chọn ứng dụng cần hoàn tiền rồi bấm "
            "Báo cáo sự cố. Kết quả hoàn tiền sẽ được gửi đến email đăng ký Google Play "
            "và hiển thị trên ứng dụng MoMo; thời gian xử lý phụ thuộc vào quy định của Google Play."
        ),
        "google_play_refund_result_location": (
            "Kết quả hoàn tiền sẽ được gửi đến email đăng ký Google Play và hiển thị "
            "trên ứng dụng MoMo. Thời gian xử lý phụ thuộc vào quy định của Google Play."
        ),
    }
    try:
        return messages[message_key]
    except KeyError as exc:
        raise ValueError(f"no customer-facing response is approved for {message_key}") from exc


def is_grounded_google_play_response(response: str, *, message_key: str) -> bool:
    lowered = response.casefold()
    if len(response) > 2_000 or any(
        phrase in lowered for phrase in ("chắc chắn", "ngay lập tức", "đảm bảo")
    ):
        return False
    if message_key == "google_play_refund_result_location":
        return all(
            term in lowered
            for term in ("email", "google play", "momo", "thời gian xử lý")
        )
    if message_key != "google_play_refund_request_steps":
        return False
    required_terms = (
        "google play",
        "lịch sử đơn đặt hàng",
        "báo cáo sự cố",
        "email",
        "momo",
        "thời gian xử lý",
    )
    return all(term in lowered for term in required_terms)


def evaluate_cashback_policy(
    request: CashbackPolicyInput,
) -> CashbackPolicyDecision:
    """Apply the customer cashback FAQ without inferring eligibility."""

    if request.reason == "unsupported_service":
        return CashbackPolicyDecision(
            action=CashbackPolicyAction.EXPLAIN_NOT_ELIGIBLE,
            message_key="cashback_not_eligible",
        )
    if request.reason == "account_limit":
        return CashbackPolicyDecision(
            action=CashbackPolicyAction.EXPLAIN_ACCOUNT_LIMIT,
            message_key="cashback_account_limit_reached",
        )
    if request.reason == "monthly_limit":
        return CashbackPolicyDecision(
            action=CashbackPolicyAction.EXPLAIN_MONTHLY_LIMIT,
            message_key="cashback_monthly_limit_reached",
        )
    if request.elapsed_hours is not None and request.elapsed_hours > 24:
        return CashbackPolicyDecision(
            action=CashbackPolicyAction.HANDOFF,
            message_key="cashback_overdue_help",
            handoff_required=True,
        )
    return CashbackPolicyDecision(
        action=CashbackPolicyAction.WAIT_WITHIN_24_HOURS,
        message_key="cashback_pending_within_24_hours",
    )


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
