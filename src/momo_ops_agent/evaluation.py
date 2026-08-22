"""Evaluation contracts for the synthetic Vietnamese MoMo golden set."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .contracts import CaseAction, CaseStatus, IntentName, SlotName, StrictModel, get_intent_contract


class GoldenTurnRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"


class ToolName(str, Enum):
    GET_TRANSACTION_STATUS = "get_transaction_status"
    GET_REFUND_STATUS = "get_refund_status"
    CREATE_SUPPORT_TICKET = "create_support_ticket"


class OutcomeName(str, Enum):
    ASK_FOR_TRANSACTION_ID = "ask_for_transaction_id"
    ANSWER_FROM_TOOL_RESULT = "answer_from_tool_result"
    CONFIRM_TRANSACTION_COMPLETED = "confirm_transaction_completed"
    CLOSE_CASE = "close_case"
    CREATE_REFUND_INVESTIGATION_TICKET = "create_refund_investigation_ticket"
    CREATE_TRANSACTION_FAILURE_TICKET = "create_transaction_failure_ticket"
    ESCALATE_COMPLETED_REFUND_DISPUTE = "escalate_completed_refund_dispute"
    EXPLAIN_PENDING_STATUS = "explain_pending_status"
    EXPLAIN_REFUND_PROCESSING = "explain_refund_processing"
    EXPLAIN_TRANSACTION_FAILED = "explain_transaction_failed"
    HANDOFF_AFTER_FAILED_TRANSACTION = "handoff_after_failed_transaction"
    HANDOFF_ON_CUSTOMER_REQUEST = "handoff_on_customer_request"
    POLICY_GUARDRAIL_HANDOFF = "policy_guardrail_handoff"
    RETRIEVE_REFUND_STATUS = "retrieve_refund_status"
    RETRIEVE_TRANSACTION_STATUS = "retrieve_transaction_status"
    RETRIEVE_TRANSACTION_STATUS_BEFORE_ANY_ACTION = (
        "retrieve_transaction_status_before_any_action"
    )
    RETRIEVE_TRANSACTION_STATUS_BEFORE_ANY_RETRY = (
        "retrieve_transaction_status_before_any_retry"
    )
    RETRIEVE_TRANSACTION_STATUS_BEFORE_CLASSIFICATION = (
        "retrieve_transaction_status_before_classification"
    )


class GoldenTurn(StrictModel):
    role: GoldenTurnRole
    text: str = Field(min_length=1, max_length=8_000)


class GoldenCase(StrictModel):
    """Expected behavior for one deterministic evaluation scenario.

    This is synthetic data authored for this project. It describes the
    expected boundary/action/outcome, not a real MoMo policy or customer case.
    """

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^momo-golden-v1-[0-9]{3}$")
    language: Literal["vi"] = "vi"
    source: Literal["human_authored_synthetic"] = "human_authored_synthetic"
    turns: tuple[GoldenTurn, ...] = Field(min_length=1)
    expected_intent: IntentName
    expected_slots: dict[SlotName, str] = Field(default_factory=dict)
    expected_action: CaseAction
    expected_tool: ToolName | None = None
    expected_status: CaseStatus
    expected_outcome: OutcomeName
    tags: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_expected_behavior(self) -> "GoldenCase":
        if self.expected_intent is IntentName.UNKNOWN:
            raise ValueError("golden set cases must target a supported intent")
        if self.turns[-1].role is not GoldenTurnRole.CUSTOMER:
            raise ValueError("the final turn must be a customer message")

        contract = get_intent_contract(self.expected_intent)
        if not contract.allows_action(self.expected_action):
            raise ValueError(
                f"{self.expected_action.value} is not allowed for "
                f"{self.expected_intent.value}"
            )

        tool_actions = {CaseAction.RETRIEVE_CONTEXT, CaseAction.EXECUTE_TOOL}
        if self.expected_action in tool_actions and self.expected_tool is None:
            raise ValueError("context/tool actions require expected_tool")
        if self.expected_action not in tool_actions and self.expected_tool is not None:
            raise ValueError("expected_tool is only valid for context/tool actions")

        if self.expected_action in tool_actions:
            missing_slots = set(contract.required_slots) - set(self.expected_slots)
            if missing_slots:
                values = ", ".join(sorted(slot.value for slot in missing_slots))
                raise ValueError(f"tool action is missing required slots: {values}")
        return self
