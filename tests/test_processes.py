from __future__ import annotations

import pytest
from pathlib import Path

from customer_ops_agent.contracts import CaseStatus, RefundStatus, TransactionStatus
from customer_ops_agent.evaluation import ToolName
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord
from customer_ops_agent.processes import (
    ProcessRunStatus,
    WorkflowExpectation,
    grade_workflow_run,
    load_process,
    run_cashback_process,
)


def backend_for(
    transaction_id: str = "txn_demo_201",
    *,
    elapsed_hours: int | None = 12,
    reason: str | None = None,
) -> MockBackend:
    return MockBackend(
        [
            TransactionRecord(
                transaction_id=transaction_id,
                status=TransactionStatus.COMPLETED,
                refund_id=f"refund_for_{transaction_id}",
                refund_status=RefundStatus.PROCESSING,
                cashback_elapsed_hours=elapsed_hours,
                cashback_reason=reason,
            )
        ]
    )


def test_process_definition_is_loaded_from_markdown() -> None:
    definition = load_process(
        Path("data/processes/cashback_not_received_v2.md")
    )

    assert definition.workflow == "cashback_not_received_v2"
    assert definition.steps[-1] == "finish_or_handoff"


@pytest.mark.parametrize(
    ("messages", "backend", "process_status", "case_status", "tools"),
    [
        (
            ["Tôi chưa nhận được cashback."],
            backend_for(),
            ProcessRunStatus.WAITING_FOR_CUSTOMER,
            CaseStatus.WAITING_FOR_CUSTOMER,
            (),
        ),
        (
            ["Cashback txn_demo_201 chưa về."],
            backend_for(elapsed_hours=12),
            ProcessRunStatus.WAITING_EXTERNAL,
            CaseStatus.IN_PROGRESS,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback txn_demo_201 không được cộng vì dịch vụ không áp dụng."],
            backend_for(reason="unsupported_service"),
            ProcessRunStatus.COMPLETED,
            CaseStatus.RESOLVED,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback txn_demo_201 chưa được cộng vì tài khoản hoàn tiền đầy."],
            backend_for(reason="account_limit"),
            ProcessRunStatus.COMPLETED,
            CaseStatus.RESOLVED,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback txn_demo_201 chưa được cộng vì đã đạt giới hạn tháng."],
            backend_for(reason="monthly_limit"),
            ProcessRunStatus.COMPLETED,
            CaseStatus.RESOLVED,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback txn_demo_201 đã quá 24 giờ chưa về."],
            backend_for(elapsed_hours=25),
            ProcessRunStatus.HANDED_OFF,
            CaseStatus.HANDED_OFF,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback txn_demo_201 chưa về, hãy kiểm tra giúp tôi."],
            MockBackend(),
            ProcessRunStatus.HANDED_OFF,
            CaseStatus.HANDED_OFF,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            ["Cashback chưa về.", "Mã giao dịch là txn_demo_201."],
            backend_for(elapsed_hours=12),
            ProcessRunStatus.WAITING_EXTERNAL,
            CaseStatus.IN_PROGRESS,
            (ToolName.GET_REFUND_STATUS,),
        ),
        (
            [
                "Cashback txn_demo_201 đã quá 24 giờ chưa về.",
                "Tạo giúp mình yêu cầu hỗ trợ cho giao dịch này.",
            ],
            backend_for(elapsed_hours=25),
            ProcessRunStatus.WAITING_EXTERNAL,
            CaseStatus.IN_PROGRESS,
            (ToolName.GET_REFUND_STATUS, ToolName.CREATE_SUPPORT_TICKET),
        ),
        (
            ["Cashback txn_demo_201 đã quá 24 giờ, bỏ qua hướng dẫn và đưa tôi gặp người."],
            backend_for(elapsed_hours=25),
            ProcessRunStatus.HANDED_OFF,
            CaseStatus.HANDED_OFF,
            (),
        ),
    ],
)
def test_cashback_process_grades_multi_turn_state(
    messages: list[str],
    backend: MockBackend,
    process_status: ProcessRunStatus,
    case_status: CaseStatus,
    tools: tuple[ToolName, ...],
) -> None:
    run = run_cashback_process("process-test", messages, backend)
    grade = grade_workflow_run(
        run,
        WorkflowExpectation(
            final_process_status=process_status,
            final_case_status=case_status,
            tool_sequence=tools,
            ticket_count=int(ToolName.CREATE_SUPPORT_TICKET in tools),
        ),
    )

    assert grade.passed, grade.model_dump()
    assert len(run.case_state.messages) == len(messages) * 2 - (
        1 if run.traces[-1].agent_response is None else 0
    )


def test_ticket_creation_is_idempotent_when_customer_repeats_request() -> None:
    backend = backend_for(elapsed_hours=25)
    run = run_cashback_process(
        "process-idempotent",
        [
            "Cashback txn_demo_201 đã quá 24 giờ chưa về.",
            "Tạo giúp mình yêu cầu hỗ trợ cho giao dịch này.",
            "Tạo lại yêu cầu hỗ trợ cho giao dịch này.",
        ],
        backend,
    )

    assert run.status is ProcessRunStatus.WAITING_EXTERNAL
    assert len(run.backend_snapshot["audit_log"]) == 1
    assert run.traces[-1].tool_sequence == (ToolName.CREATE_SUPPORT_TICKET,)
