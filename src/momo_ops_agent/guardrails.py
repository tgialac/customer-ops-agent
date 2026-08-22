"""Deterministic input/output guardrails for the customer-ops runtime."""

from __future__ import annotations

import re
from enum import Enum

from .contracts import CaseAction, IntentName, StrictModel
from .policies import (
    BANK_TRANSFER_POLICY_SOURCE,
    CASHBACK_POLICY_SOURCE,
    GOOGLE_PLAY_REFUND_POLICY_SOURCE,
    is_grounded_bank_transfer_response,
    is_grounded_cashback_response,
    is_grounded_google_play_response,
)


class GuardrailStage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class GuardrailResult(StrictModel):
    stage: GuardrailStage
    passed: bool
    reason: str | None = None


_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|show|print)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
        re.I,
    ),
    re.compile(r"\bbỏ qua\s+(?:mọi\s+)?hướng dẫn\b", re.I),
    re.compile(r"\btiết lộ\s+(?:system|developer)?\s*prompt\b", re.I),
    re.compile(
        r"\b(?:đóng vai|giả làm)\s+(?:system|developer|quản trị viên)\b",
        re.I,
    ),
)


def check_input(text: str) -> GuardrailResult:
    """Reject malformed or instruction-hijacking customer input."""

    if not text.strip():
        return GuardrailResult(
            stage=GuardrailStage.INPUT,
            passed=False,
            reason="blank_customer_message",
        )
    if len(text) > 8_000:
        return GuardrailResult(
            stage=GuardrailStage.INPUT,
            passed=False,
            reason="customer_message_too_long",
        )
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                stage=GuardrailStage.INPUT,
                passed=False,
                reason="prompt_injection_pattern",
            )
    return GuardrailResult(stage=GuardrailStage.INPUT, passed=True)


def check_output(
    *,
    intent: IntentName,
    action: CaseAction,
    response: str,
    policy_source: str | None,
    policy_message_key: str | None = None,
    return_destination: str | None = None,
) -> GuardrailResult:
    """Allow only policy-bound output for the source-backed workflow."""

    if intent is not IntentName.BANK_TRANSFER_NOT_RECEIVED:
        if not (
            intent is IntentName.MISSING_REFUND
            and policy_source in {
                CASHBACK_POLICY_SOURCE,
                GOOGLE_PLAY_REFUND_POLICY_SOURCE,
            }
        ):
            return GuardrailResult(stage=GuardrailStage.OUTPUT, passed=True)

    if action is CaseAction.HANDOFF:
        passed = response == "handoff"
        reason = None if passed else "handoff_response_mismatch"
    elif action is CaseAction.ASK_CLARIFICATION:
        passed = response == "ask_for_transaction_id"
        reason = None if passed else "clarification_response_mismatch"
    elif action is CaseAction.ANSWER and policy_source == CASHBACK_POLICY_SOURCE:
        passed = (
            policy_message_key is not None
            and is_grounded_cashback_response(response, message_key=policy_message_key)
        )
        reason = None if passed else "cashback_answer_not_bound_to_policy"
    elif action is CaseAction.ANSWER and policy_source == GOOGLE_PLAY_REFUND_POLICY_SOURCE:
        passed = (
            policy_message_key is not None
            and is_grounded_google_play_response(response, message_key=policy_message_key)
        )
        reason = None if passed else "google_play_answer_not_bound_to_policy"
    elif action is CaseAction.ANSWER:
        passed = (
            policy_source == BANK_TRANSFER_POLICY_SOURCE
            and policy_message_key is not None
            and is_grounded_bank_transfer_response(
                response,
                message_key=policy_message_key,
                return_destination=return_destination,
            )
        )
        reason = None if passed else "answer_not_bound_to_approved_policy"
    else:
        passed = False
        reason = "unsupported_customer_facing_action"

    return GuardrailResult(
        stage=GuardrailStage.OUTPUT,
        passed=passed,
        reason=reason,
    )
