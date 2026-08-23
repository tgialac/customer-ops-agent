"""Typed contracts for case state, intent routing, and context boundaries.

The state is deliberately kept independent from any LLM/provider.  It can be
serialized to JSON, passed between workflow steps, and used as the source of
truth for tool authorization and evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class StrictModel(BaseModel):
    """Reject unknown fields so external/LLM output cannot silently drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CaseChannel(str, Enum):
    CHAT = "chat"
    BACKOFFICE = "backoffice"


class CaseStatus(str, Enum):
    NEW = "new"
    TRIAGING = "triaging"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_CUSTOMER = "waiting_for_customer"
    RESOLVED = "resolved"
    HANDED_OFF = "handed_off"
    CLOSED = "closed"


class CaseAction(str, Enum):
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    RETRIEVE_CONTEXT = "retrieve_context"
    EXECUTE_TOOL = "execute_tool"
    HANDOFF = "handoff"
    CLOSE = "close"


ACTIONS = tuple(CaseAction)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FundingSource(str, Enum):
    WALLET = "wallet"
    LINKED_BANK = "linked_bank"
    UNKNOWN = "unknown"


class IntentName(str, Enum):
    UNKNOWN = "unknown"
    MISSING_REFUND = "missing_refund"
    TRANSACTION_PENDING = "transaction_pending"
    TRANSACTION_FAILED = "transaction_failed"
    BANK_TRANSFER_NOT_RECEIVED = "bank_transfer_not_received"


class ContextScope(str, Enum):
    ACCOUNT = "account"
    TRANSACTION = "transaction"
    REFUND = "refund"
    FRAUD = "fraud"


class SlotName(str, Enum):
    TRANSACTION_ID = "transaction_id"
    REFUND_ID = "refund_id"


class MessageRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    HUMAN = "human"
    TOOL = "tool"
    SYSTEM = "system"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"


class RefundStatus(str, Enum):
    REQUESTED = "requested"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversationMessage(StrictModel):
    message_id: str = Field(default_factory=lambda: _new_id("msg"), min_length=1)
    role: MessageRole
    content: str = Field(min_length=1, max_length=8_000)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class IntentAlternative(StrictModel):
    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)


class IntentPrediction(StrictModel):
    """The classifier result used for routing, never an unvalidated string."""

    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["classifier", "rule", "human"] = "classifier"
    model_version: str | None = Field(default=None, min_length=1)
    missing_slots: tuple[SlotName, ...] = ()
    alternatives: tuple[IntentAlternative, ...] = ()


class IntentContract(StrictModel):
    """Policy contract for one intent.

    It defines the maximum context and action surface available to a routed
    intent.  The agent may use less context/actions, but not more.
    """

    intent: IntentName
    description: str = Field(min_length=1)
    required_slots: tuple[SlotName, ...] = ()
    allowed_context: tuple[ContextScope, ...] = ()
    allowed_actions: tuple[CaseAction, ...]
    risk_level: RiskLevel
    minimum_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    def allows_action(self, action: CaseAction) -> bool:
        return action in self.allowed_actions

    def allows_context(self, scope: ContextScope) -> bool:
        return scope in self.allowed_context


_COMMON_SUPPORT_ACTIONS = (
    CaseAction.ANSWER,
    CaseAction.ASK_CLARIFICATION,
    CaseAction.RETRIEVE_CONTEXT,
    CaseAction.EXECUTE_TOOL,
    CaseAction.HANDOFF,
    CaseAction.CLOSE,
)

INTENT_CATALOG: Mapping[IntentName, IntentContract] = {
    IntentName.UNKNOWN: IntentContract(
        intent=IntentName.UNKNOWN,
        description="The request cannot yet be mapped to a supported customer intent.",
        allowed_actions=(CaseAction.ASK_CLARIFICATION, CaseAction.HANDOFF),
        risk_level=RiskLevel.MEDIUM,
        minimum_confidence=0.0,
    ),
    IntentName.MISSING_REFUND: IntentContract(
        intent=IntentName.MISSING_REFUND,
        description="A customer reports that an expected refund has not arrived.",
        required_slots=(SlotName.TRANSACTION_ID,),
        allowed_context=(ContextScope.TRANSACTION, ContextScope.REFUND),
        allowed_actions=_COMMON_SUPPORT_ACTIONS,
        risk_level=RiskLevel.MEDIUM,
    ),
    IntentName.TRANSACTION_PENDING: IntentContract(
        intent=IntentName.TRANSACTION_PENDING,
        description="A customer asks about a transaction that is still pending.",
        required_slots=(SlotName.TRANSACTION_ID,),
        allowed_context=(ContextScope.TRANSACTION,),
        allowed_actions=_COMMON_SUPPORT_ACTIONS,
        risk_level=RiskLevel.MEDIUM,
    ),
    IntentName.TRANSACTION_FAILED: IntentContract(
        intent=IntentName.TRANSACTION_FAILED,
        description="A customer reports that a transaction failed.",
        required_slots=(SlotName.TRANSACTION_ID,),
        allowed_context=(ContextScope.TRANSACTION,),
        allowed_actions=_COMMON_SUPPORT_ACTIONS,
        risk_level=RiskLevel.MEDIUM,
    ),
    IntentName.BANK_TRANSFER_NOT_RECEIVED: IntentContract(
        intent=IntentName.BANK_TRANSFER_NOT_RECEIVED,
        description=(
            "A customer says a MoMo-to-bank transfer was debited but the "
            "beneficiary has not received the money."
        ),
        required_slots=(SlotName.TRANSACTION_ID,),
        allowed_context=(ContextScope.TRANSACTION,),
        allowed_actions=_COMMON_SUPPORT_ACTIONS,
        risk_level=RiskLevel.MEDIUM,
    ),
}


def get_intent_contract(intent: IntentName) -> IntentContract:
    return INTENT_CATALOG[intent]


class TransactionSnapshot(StrictModel):
    transaction_id: str = Field(min_length=1)
    status: TransactionStatus
    amount_minor: int = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    occurred_at: datetime | None = None
    funding_source: FundingSource = FundingSource.UNKNOWN
    elapsed_working_days: int | None = Field(default=None, ge=0)
    return_elapsed_working_days: int | None = Field(default=None, ge=0)
    wrong_details_reported: bool = False


class RefundSnapshot(StrictModel):
    refund_id: str = Field(min_length=1)
    status: RefundStatus
    amount_minor: int = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    requested_at: datetime | None = None
    expected_by: datetime | None = None
    cashback_elapsed_hours: int | None = Field(default=None, ge=0)
    cashback_reason: str | None = None


class ContextSnapshot(StrictModel):
    """Intent-gated, typed context; no raw PII belongs in this object."""

    scopes: tuple[ContextScope, ...] = ()
    transaction: TransactionSnapshot | None = None
    refund: RefundSnapshot | None = None

    @model_validator(mode="after")
    def context_must_match_declared_scopes(self) -> "ContextSnapshot":
        if self.transaction is not None and ContextScope.TRANSACTION not in self.scopes:
            raise ValueError("transaction context requires the transaction scope")
        if self.refund is not None and ContextScope.REFUND not in self.scopes:
            raise ValueError("refund context requires the refund scope")
        return self


class CaseState(StrictModel):
    """The durable state passed between triage, tools, and response steps."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: Literal[1] = 1
    case_id: str = Field(default_factory=lambda: _new_id("case"), min_length=1)
    customer_ref: str = Field(min_length=1, description="Non-PII customer reference")
    channel: CaseChannel = CaseChannel.CHAT
    status: CaseStatus = CaseStatus.NEW
    risk_level: RiskLevel = RiskLevel.LOW
    intent: IntentPrediction | None = None
    context: ContextSnapshot = Field(default_factory=ContextSnapshot)
    collected_slots: dict[SlotName, str] = Field(default_factory=dict)
    messages: list[ConversationMessage] = Field(default_factory=list)
    input_guardrail_failures: int = Field(default=0, ge=0)
    output_guardrail_failures: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    handoff_reason: str | None = Field(default=None, min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def intent_must_bound_context(self) -> "CaseState":
        if self.intent is None:
            return self
        contract = get_intent_contract(self.intent.intent)
        unsupported = set(self.context.scopes) - set(contract.allowed_context)
        if unsupported:
            values = ", ".join(sorted(scope.value for scope in unsupported))
            raise ValueError(f"context scopes not allowed for {self.intent.intent.value}: {values}")
        return self

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamps_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    def accept_intent(self, prediction: IntentPrediction) -> "CaseState":
        contract = get_intent_contract(prediction.intent)
        if prediction.confidence < contract.minimum_confidence:
            raise ValueError(
                f"confidence {prediction.confidence:.3f} is below the "
                f"{contract.minimum_confidence:.3f} threshold for {prediction.intent.value}"
            )
        next_risk = _higher_risk(self.risk_level, contract.risk_level)
        return self._validated_copy(
            intent=prediction,
            risk_level=next_risk,
            updated_at=_utc_now(),
        )

    def transition_to(self, status: CaseStatus, *, reason: str | None = None) -> "CaseState":
        if status == self.status:
            return self
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid case transition: {self.status.value} -> {status.value}")
        if status == CaseStatus.HANDED_OFF and not reason:
            raise ValueError("handoff requires a reason")
        return self._validated_copy(
            status=status,
            handoff_reason=(
                reason if status == CaseStatus.HANDED_OFF else self.handoff_reason
            ),
            updated_at=_utc_now(),
        )

    def with_context(self, context: ContextSnapshot) -> "CaseState":
        return self._validated_copy(context=context, updated_at=_utc_now())

    def add_message(self, message: ConversationMessage) -> "CaseState":
        return self._validated_copy(
            messages=[*self.messages, message],
            updated_at=_utc_now(),
        )

    def record_guardrail_failure(
        self, stage: Literal["input", "output"]
    ) -> "CaseState":
        updates = {
            "input_guardrail_failures": self.input_guardrail_failures,
            "output_guardrail_failures": self.output_guardrail_failures,
            "updated_at": _utc_now(),
        }
        key = f"{stage}_guardrail_failures"
        updates[key] = updates[key] + 1
        return self._validated_copy(**updates)

    def _validated_copy(self, **updates: object) -> "CaseState":
        values = self.model_dump()
        values.update(updates)
        return type(self).model_validate(values)


_ALLOWED_TRANSITIONS: Mapping[CaseStatus, frozenset[CaseStatus]] = {
    CaseStatus.NEW: frozenset({CaseStatus.TRIAGING, CaseStatus.HANDED_OFF}),
    CaseStatus.TRIAGING: frozenset(
        {CaseStatus.IN_PROGRESS, CaseStatus.WAITING_FOR_CUSTOMER, CaseStatus.HANDED_OFF}
    ),
    CaseStatus.IN_PROGRESS: frozenset(
        {
            CaseStatus.WAITING_FOR_CUSTOMER,
            CaseStatus.RESOLVED,
            CaseStatus.HANDED_OFF,
        }
    ),
    CaseStatus.WAITING_FOR_CUSTOMER: frozenset(
        {CaseStatus.IN_PROGRESS, CaseStatus.HANDED_OFF}
    ),
    CaseStatus.RESOLVED: frozenset({CaseStatus.CLOSED, CaseStatus.IN_PROGRESS}),
    CaseStatus.HANDED_OFF: frozenset(
        {CaseStatus.IN_PROGRESS, CaseStatus.RESOLVED, CaseStatus.CLOSED}
    ),
    CaseStatus.CLOSED: frozenset(),
}


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def _higher_risk(left: RiskLevel, right: RiskLevel) -> RiskLevel:
    return left if _RISK_ORDER[left] >= _RISK_ORDER[right] else right
