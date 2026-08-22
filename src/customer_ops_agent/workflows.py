"""Explicit workflow scopes used to keep policy and tools from bleeding together."""

from __future__ import annotations

from enum import Enum

from .contracts import SlotName, StrictModel


class WorkflowName(str, Enum):
    BANK_TRANSFER_NOT_RECEIVED_V1 = "bank_transfer_not_received_v1"
    CASHBACK_NOT_RECEIVED_V1 = "cashback_not_received_v1"
    GOOGLE_PLAY_REFUND_V1 = "google_play_refund_v1"


class WorkflowSpec(StrictModel):
    workflow: WorkflowName
    customer_problem: str
    entry_condition: str
    included_scenarios: tuple[str, ...]
    excluded_scenarios: tuple[str, ...]
    required_slots: tuple[SlotName, ...]
    terminal_outcomes: tuple[str, ...]
    handoff_conditions: tuple[str, ...]
    source_urls: tuple[str, ...]


BANK_TRANSFER_NOT_RECEIVED_V1 = WorkflowSpec(
    workflow=WorkflowName.BANK_TRANSFER_NOT_RECEIVED_V1,
    customer_problem=(
        "A customer says a MoMo-to-bank transfer was debited but the beneficiary "
        "has not received the money."
    ),
    entry_condition=(
        "The customer is asking about a transfer from MoMo to a bank account or "
        "bank card, not a MoMo-to-MoMo transfer, top-up, withdrawal, cashback, "
        "merchant refund, or Google Play purchase."
    ),
    included_scenarios=(
        "pending transfer awaiting bank reconciliation",
        "successful transfer with delayed beneficiary-bank posting",
        "failed transfer with money returning to MoMo or the funding bank",
        "wrong bank details requiring support-led recovery",
        "missing transaction ID requiring clarification",
    ),
    excluded_scenarios=(
        "MoMo-to-MoMo transfer to an unregistered recipient",
        "bank top-up or withdrawal pending",
        "promotional cashback not received",
        "merchant Payment API refund",
        "Google Play app-purchase refund",
    ),
    required_slots=(SlotName.TRANSACTION_ID,),
    terminal_outcomes=(
        "explain_pending_and_wait",
        "explain_delayed_beneficiary_posting",
        "explain_failed_transfer_return",
        "request_transaction_id",
        "create_support_ticket",
        "handoff_for_wrong_details_or_overdue_case",
    ),
    handoff_conditions=(
        "the public policy window has passed without a final status",
        "the customer reports wrong recipient or wrong bank details",
        "the customer asks for a case outside this workflow",
        "a tool or answer guardrail fails",
    ),
    source_urls=(
        "https://www.momo.vn/hoi-dap/vi-sao-tai-khoan-da-bi-tru-tien-nhung-tai-khoan-ngan-hang-nguoi-nhan-chua-nhan-duoc",
        "https://www.momo.vn/hoi-dap/tai-khoan-bi-tru-tien-nhung-giao-dich-dang-cho-xu-ly",
        "https://www.momo.vn/hoi-dap/toi-co-the-chuyen-tien-cho-nguoi-chua-co-tai-khoan-momo-khong",
    ),
)


CASHBACK_NOT_RECEIVED_V1 = WorkflowSpec(
    workflow=WorkflowName.CASHBACK_NOT_RECEIVED_V1,
    customer_problem=(
        "A customer says an eligible MoMo cashback has not appeared after a payment."
    ),
    entry_condition=(
        "The customer is asking about promotional cashback or the MoMo cashback account, "
        "not a merchant refund, bank-transfer reversal, or Google Play refund."
    ),
    included_scenarios=(
        "cashback still within the public 24-hour window",
        "cashback overdue and requiring Help",
        "service outside the cashback eligibility list",
        "cashback account balance limit reached",
        "monthly cashback limit reached",
        "missing transaction ID requiring clarification",
    ),
    excluded_scenarios=(
        "merchant Payment API refund",
        "bank-transfer reversal",
        "Google Play purchase refund",
        "cashback policy facts not verified by the refund-status tool",
    ),
    required_slots=(SlotName.TRANSACTION_ID,),
    terminal_outcomes=(
        "wait_within_24_hours",
        "explain_cashback_not_eligible",
        "explain_cashback_account_limit",
        "explain_cashback_monthly_limit",
        "handoff_after_cashback_window",
        "request_transaction_id",
    ),
    handoff_conditions=(
        "the public 24-hour window has passed without cashback",
        "the refund-status tool fails or returns no verified cashback facts",
        "an answer guardrail fails",
    ),
    source_urls=(
        "https://www.momo.vn/hoi-dap/tai-sao-toi-khong-duoc-hoan-tien-khi-thanh-toan-dich-vu-nay",
    ),
)


GOOGLE_PLAY_REFUND_V1 = WorkflowSpec(
    workflow=WorkflowName.GOOGLE_PLAY_REFUND_V1,
    customer_problem=(
        "A customer wants to request a refund for an app purchased through Google Play."
    ),
    entry_condition=(
        "The customer explicitly mentions Google Play and asks about an app refund."
    ),
    included_scenarios=(
        "requesting a Google Play app refund",
        "asking where to submit the refund request",
        "asking where the refund result will appear",
    ),
    excluded_scenarios=(
        "merchant Payment API refund",
        "cashback not received",
        "bank-transfer reversal",
        "refund timing guarantees",
    ),
    required_slots=(),
    terminal_outcomes=("provide_google_play_refund_steps",),
    handoff_conditions=(
        "the customer needs a case-specific status not covered by the public FAQ",
        "an answer guardrail fails",
    ),
    source_urls=(
        "https://www.momo.vn/hoi-dap/toi-muon-hoan-tien-da-giao-dich-cho-ung-dung-da-mua",
    ),
)


WORKFLOW_CATALOG = {
    WorkflowName.BANK_TRANSFER_NOT_RECEIVED_V1: BANK_TRANSFER_NOT_RECEIVED_V1,
    WorkflowName.CASHBACK_NOT_RECEIVED_V1: CASHBACK_NOT_RECEIVED_V1,
    WorkflowName.GOOGLE_PLAY_REFUND_V1: GOOGLE_PLAY_REFUND_V1,
}


def get_workflow(workflow: WorkflowName) -> WorkflowSpec:
    return WORKFLOW_CATALOG[workflow]
