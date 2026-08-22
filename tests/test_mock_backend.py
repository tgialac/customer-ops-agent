from customer_ops_agent.contracts import TransactionStatus
from customer_ops_agent.evaluation import ToolName
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord


def test_support_ticket_mutates_state_once_and_is_idempotent() -> None:
    backend = MockBackend(
        [TransactionRecord(transaction_id="txn_demo_900", status=TransactionStatus.FAILED)]
    )

    first = backend.create_support_ticket("txn_demo_900", "investigate_failure")
    second = backend.create_support_ticket("txn_demo_900", "investigate_failure")
    snapshot = backend.snapshot()

    assert first.tool_name is ToolName.CREATE_SUPPORT_TICKET
    assert first.success is True
    assert first.idempotent_replay is False
    assert second.data["ticket_id"] == first.data["ticket_id"]
    assert second.idempotent_replay is True
    assert snapshot["state_version"] == 2
    assert len(snapshot["audit_log"]) == 1


def test_missing_transaction_returns_deterministic_error() -> None:
    result = MockBackend().get_transaction_status("txn_missing")

    assert result.success is False
    assert result.error_code == "transaction_not_found"
    assert result.tool_name is ToolName.GET_TRANSACTION_STATUS
