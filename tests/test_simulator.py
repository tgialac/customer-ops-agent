from __future__ import annotations

from customer_ops_agent.contracts import RefundStatus, TransactionStatus
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord
from customer_ops_agent.processes import ProcessRunStatus
from customer_ops_agent.simulator import (
    LLMUserSimulator,
    ScriptedUserSimulator,
    UserSimulationGoal,
    UserSimulationResponse,
    run_simulated_cashback_process,
)


def backend_for_cashback() -> MockBackend:
    return MockBackend(
        [
            TransactionRecord(
                transaction_id="txn_demo_201",
                status=TransactionStatus.COMPLETED,
                refund_id="refund_for_txn_demo_201",
                refund_status=RefundStatus.PROCESSING,
                cashback_elapsed_hours=12,
            )
        ]
    )


def test_scripted_simulator_reacts_on_one_persistent_session() -> None:
    result = run_simulated_cashback_process(
        "sim-scripted-001",
        backend_for_cashback(),
        ScriptedUserSimulator(
            ["Cashback của tôi chưa về.", "Mã giao dịch là txn_demo_201."]
        ),
        UserSimulationGoal(
            scenario="Cashback is within the normal 24-hour window.",
            objective="Provide the transaction ID when asked.",
        ),
    )

    assert result.stop_reason == "simulator_done"
    assert result.run.status is ProcessRunStatus.WAITING_EXTERNAL
    assert [turn.role for turn in result.transcript] == [
        "customer",
        "agent",
        "customer",
        "agent",
    ]
    assert result.run.traces[1].tool_sequence


def test_llm_simulator_provider_receives_agent_context() -> None:
    requests = []

    def provider(request):
        requests.append(request)
        if not request.transcript:
            return UserSimulationResponse(message="Cashback của tôi chưa về.")
        assert request.last_agent_message is not None
        return UserSimulationResponse(
            message="Mã giao dịch là txn_demo_201.",
            done=True,
        )

    result = run_simulated_cashback_process(
        "sim-llm-provider-001",
        backend_for_cashback(),
        LLMUserSimulator(provider),
        UserSimulationGoal(
            scenario="Cashback is within the normal 24-hour window.",
            objective="Provide the transaction ID when asked.",
            max_turns=3,
        ),
    )

    assert len(requests) == 2
    assert requests[1].last_agent_message is not None
    assert result.run.status is ProcessRunStatus.WAITING_EXTERNAL
