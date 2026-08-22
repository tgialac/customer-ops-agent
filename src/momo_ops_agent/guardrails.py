"""Deterministic input/output guardrails for the customer-ops runtime."""

from __future__ import annotations

import re
from enum import Enum

from .contracts import CaseAction, IntentName, StrictModel
from .policies import BANK_TRANSFER_POLICY_SOURCE, is_approved_bank_transfer_response


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
) -> GuardrailResult:
    """Allow only policy-bound output for the source-backed workflow."""

    if intent is not IntentName.BANK_TRANSFER_NOT_RECEIVED:
        return GuardrailResult(stage=GuardrailStage.OUTPUT, passed=True)

    if action is CaseAction.HANDOFF:
        passed = response == "handoff"
        reason = None if passed else "handoff_response_mismatch"
    elif action is CaseAction.ASK_CLARIFICATION:
        passed = response == "ask_for_transaction_id"
        reason = None if passed else "clarification_response_mismatch"
    elif action is CaseAction.ANSWER:
        passed = (
            policy_source == BANK_TRANSFER_POLICY_SOURCE
            and is_approved_bank_transfer_response(response)
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
