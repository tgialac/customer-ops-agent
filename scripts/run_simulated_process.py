"""Run the cashback process with a scripted or OpenAI user simulator."""

from __future__ import annotations

import argparse
import json

from customer_ops_agent.contracts import RefundStatus, TransactionStatus
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord
from customer_ops_agent.simulator import (
    LLMUserSimulator,
    ScriptedUserSimulator,
    UserSimulationGoal,
    run_simulated_cashback_process,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=("scripted", "openai"), default="scripted")
    parser.add_argument("--model")
    args = parser.parse_args()

    backend = MockBackend(
        [
            TransactionRecord(
                transaction_id="txn_demo_201",
                status=TransactionStatus.COMPLETED,
                refund_id="refund_for_txn_demo_201",
                refund_status=RefundStatus.PROCESSING,
                cashback_elapsed_hours=25,
            )
        ]
    )
    simulator = (
        LLMUserSimulator.from_environment(args.model)
        if args.simulator == "openai"
        else ScriptedUserSimulator(
            [
                "Cashback của tôi chưa về.",
                "Mã giao dịch là txn_demo_201.",
                "Tạo giúp mình yêu cầu hỗ trợ cho giao dịch này.",
            ]
        )
    )
    result = run_simulated_cashback_process(
        "simulated-process-demo",
        backend,
        simulator,
        UserSimulationGoal(
            scenario=(
                "The cashback is overdue. The customer has the transaction ID "
                "but only provides it after the agent asks."
            ),
            objective="Get the case investigated by support.",
        ),
    )
    print(
        json.dumps(
            {
                "stop_reason": result.stop_reason,
                "process_status": result.run.status.value,
                "case_status": result.run.case_state.status.value,
                "transcript": result.transcript,
                "tool_sequence": [
                    tool.value
                    for trace in result.run.traces
                    for tool in trace.tool_sequence
                ],
                "backend_audit_log": result.run.backend_snapshot["audit_log"],
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.model_dump(mode="json"),
        )
    )


if __name__ == "__main__":
    main()
