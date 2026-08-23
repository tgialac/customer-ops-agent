"""Show a multi-turn cashback process and its evolving plan."""

from __future__ import annotations

import json

from customer_ops_agent.contracts import RefundStatus, TransactionStatus
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord
from customer_ops_agent.processes import run_cashback_process


def main() -> None:
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
    run = run_cashback_process(
        "cashback-process-demo",
        [
            "Cashback của tôi chưa về.",
            "Mã giao dịch là txn_demo_201.",
            "Tạo giúp mình yêu cầu hỗ trợ cho giao dịch này.",
        ],
        backend,
    )
    print(
        json.dumps(
            {
                "status": run.status.value,
                "case_status": run.case_state.status.value,
                "current_step": run.plan.current_step,
                "steps": [step.model_dump(mode="json") for step in run.plan.steps],
                "turns": [trace.model_dump(mode="json") for trace in run.traces],
                "backend_audit_log": run.backend_snapshot["audit_log"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
