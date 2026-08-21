"""A deterministic, stateful fintech backend for offline agent evaluation."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import Field

from .contracts import RefundStatus, StrictModel, TransactionStatus
from .evaluation import ToolName


class TransactionRecord(StrictModel):
    transaction_id: str = Field(min_length=1)
    status: TransactionStatus
    amount_minor: int = Field(default=100_000, ge=0)
    currency: str = Field(default="VND", pattern=r"^[A-Z]{3}$")
    refund_id: str | None = None
    refund_status: RefundStatus | None = None
    ticket_id: str | None = None


class ToolResult(StrictModel):
    tool_name: ToolName
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    state_version: int = Field(ge=1)
    idempotent_replay: bool = False


class MockBackend:
    """Small state machine with realistic read/write tool behavior.

    Reads are deterministic. The support-ticket write is idempotent and
    increments the state version exactly once for a new ticket.
    """

    def __init__(self, transactions: Iterable[TransactionRecord] = ()) -> None:
        self._transactions = {item.transaction_id: item for item in transactions}
        self._state_version = 1
        self._ticket_sequence = 0
        self._audit_log: list[dict[str, Any]] = []

    def get_transaction_status(self, transaction_id: str) -> ToolResult:
        record = self._transactions.get(transaction_id)
        if record is None:
            return self._not_found(ToolName.GET_TRANSACTION_STATUS, transaction_id)
        return ToolResult(
            tool_name=ToolName.GET_TRANSACTION_STATUS,
            success=True,
            data={
                "transaction_id": record.transaction_id,
                "status": record.status.value,
                "amount_minor": record.amount_minor,
                "currency": record.currency,
            },
            state_version=self._state_version,
        )

    def get_refund_status(self, transaction_id: str) -> ToolResult:
        record = self._transactions.get(transaction_id)
        if record is None:
            return self._not_found(ToolName.GET_REFUND_STATUS, transaction_id)
        if record.refund_id is None or record.refund_status is None:
            return ToolResult(
                tool_name=ToolName.GET_REFUND_STATUS,
                success=False,
                error_code="refund_not_found",
                data={"transaction_id": transaction_id},
                state_version=self._state_version,
            )
        return ToolResult(
            tool_name=ToolName.GET_REFUND_STATUS,
            success=True,
            data={
                "transaction_id": record.transaction_id,
                "refund_id": record.refund_id,
                "refund_status": record.refund_status.value,
                "amount_minor": record.amount_minor,
                "currency": record.currency,
            },
            state_version=self._state_version,
        )

    def create_support_ticket(self, transaction_id: str, reason: str) -> ToolResult:
        record = self._transactions.get(transaction_id)
        if record is None:
            return self._not_found(ToolName.CREATE_SUPPORT_TICKET, transaction_id)
        if record.ticket_id is not None:
            return ToolResult(
                tool_name=ToolName.CREATE_SUPPORT_TICKET,
                success=True,
                data={
                    "transaction_id": transaction_id,
                    "ticket_id": record.ticket_id,
                    "reason": reason,
                },
                state_version=self._state_version,
                idempotent_replay=True,
            )

        self._ticket_sequence += 1
        ticket_id = f"ticket_demo_{self._ticket_sequence:03d}"
        self._transactions[transaction_id] = record.model_copy(update={"ticket_id": ticket_id})
        self._state_version += 1
        self._audit_log.append(
            {
                "event": "support_ticket_created",
                "transaction_id": transaction_id,
                "ticket_id": ticket_id,
                "reason": reason,
                "state_version": self._state_version,
            }
        )
        return ToolResult(
            tool_name=ToolName.CREATE_SUPPORT_TICKET,
            success=True,
            data={
                "transaction_id": transaction_id,
                "ticket_id": ticket_id,
                "reason": reason,
            },
            state_version=self._state_version,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state_version": self._state_version,
            "transactions": {
                key: value.model_dump(mode="json")
                for key, value in self._transactions.items()
            },
            "audit_log": list(self._audit_log),
        }

    def _not_found(self, tool_name: ToolName, transaction_id: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error_code="transaction_not_found",
            data={"transaction_id": transaction_id},
            state_version=self._state_version,
        )
