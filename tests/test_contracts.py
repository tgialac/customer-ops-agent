from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from momo_ops_agent.contracts import (
    CaseAction,
    CaseState,
    CaseStatus,
    ContextScope,
    ContextSnapshot,
    ConversationMessage,
    IntentName,
    IntentPrediction,
    MessageRole,
    RefundSnapshot,
    RefundStatus,
    TransactionSnapshot,
    TransactionStatus,
    get_intent_contract,
)


def _new_case() -> CaseState:
    return CaseState(customer_ref="customer_test_001")


def test_missing_refund_contract_is_explicit_about_context_and_actions() -> None:
    contract = get_intent_contract(IntentName.MISSING_REFUND)

    assert ContextScope.TRANSACTION in contract.allowed_context
    assert ContextScope.REFUND in contract.allowed_context
    assert contract.allows_action(CaseAction.EXECUTE_TOOL)
    assert contract.required_slots == ("transaction_id",)


def test_case_accepts_confident_intent_and_intent_gated_context() -> None:
    case = _new_case().accept_intent(
        IntentPrediction(intent=IntentName.MISSING_REFUND, confidence=0.93)
    )
    context = ContextSnapshot(
        scopes=(ContextScope.TRANSACTION, ContextScope.REFUND),
        transaction=TransactionSnapshot(
            transaction_id="txn_test_001",
            status=TransactionStatus.COMPLETED,
            amount_minor=100_000,
            currency="VND",
        ),
        refund=RefundSnapshot(
            refund_id="refund_test_001",
            status=RefundStatus.PROCESSING,
            amount_minor=100_000,
            currency="VND",
        ),
    )

    case = case.with_context(context)

    assert case.intent is not None
    assert case.intent.intent is IntentName.MISSING_REFUND
    assert case.context.refund is not None
    assert case.risk_level.value == "medium"


def test_low_confidence_intent_cannot_be_accepted() -> None:
    with pytest.raises(ValueError, match="below"):
        _new_case().accept_intent(
            IntentPrediction(intent=IntentName.MISSING_REFUND, confidence=0.62)
        )


def test_context_outside_intent_boundary_is_rejected() -> None:
    case = _new_case().accept_intent(
        IntentPrediction(intent=IntentName.TRANSACTION_PENDING, confidence=0.91)
    )
    fraud_context = ContextSnapshot(scopes=(ContextScope.FRAUD,))

    with pytest.raises(ValidationError, match="not allowed"):
        case.with_context(fraud_context)


def test_context_data_requires_declared_scope() -> None:
    with pytest.raises(ValidationError, match="requires the transaction scope"):
        ContextSnapshot(
            transaction=TransactionSnapshot(
                transaction_id="txn_test_001",
                status=TransactionStatus.PENDING,
                amount_minor=1,
                currency="VND",
            )
        )


def test_state_transitions_are_explicit_and_immutable_style() -> None:
    case = _new_case()
    triaging = case.transition_to(CaseStatus.TRIAGING)
    handed_off = triaging.transition_to(
        CaseStatus.HANDED_OFF, reason="intent confidence below threshold"
    )

    assert case.status is CaseStatus.NEW
    assert triaging.status is CaseStatus.TRIAGING
    assert handed_off.status is CaseStatus.HANDED_OFF
    assert handed_off.handoff_reason == "intent confidence below threshold"

    with pytest.raises(ValueError, match="invalid case transition"):
        case.transition_to(CaseStatus.CLOSED)


def test_case_serializes_to_json_with_versioned_schema() -> None:
    case = _new_case().add_message(
        ConversationMessage(role=MessageRole.CUSTOMER, content="Hoàn tiền của tôi chưa về.")
    )
    payload = case.model_dump(mode="json")
    schema = CaseState.model_json_schema()

    assert payload["schema_version"] == 1
    assert payload["messages"][0]["role"] == "customer"
    assert schema["properties"]["schema_version"]["const"] == 1


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ConversationMessage(
            role=MessageRole.CUSTOMER,
            content="hello",
            created_at=datetime(2026, 8, 21),
        )
