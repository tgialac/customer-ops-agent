"""Explicit workflow scopes used to keep policy and tools from bleeding together."""

from __future__ import annotations

from enum import Enum

from .contracts import SlotName, StrictModel


class WorkflowName(str, Enum):
    BANK_TRANSFER_NOT_RECEIVED_V1 = "bank_transfer_not_received_v1"


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


WORKFLOW_CATALOG = {
    WorkflowName.BANK_TRANSFER_NOT_RECEIVED_V1: BANK_TRANSFER_NOT_RECEIVED_V1,
}


def get_workflow(workflow: WorkflowName) -> WorkflowSpec:
    return WORKFLOW_CATALOG[workflow]
