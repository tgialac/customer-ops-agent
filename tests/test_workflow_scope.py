from momo_ops_agent.contracts import SlotName
from momo_ops_agent.workflows import (
    BANK_TRANSFER_NOT_RECEIVED_V1,
    CASHBACK_NOT_RECEIVED_V1,
    WorkflowName,
    get_workflow,
)


def test_first_workflow_is_narrow_and_source_backed() -> None:
    workflow = get_workflow(WorkflowName.BANK_TRANSFER_NOT_RECEIVED_V1)

    assert workflow is BANK_TRANSFER_NOT_RECEIVED_V1
    assert workflow.required_slots == (SlotName.TRANSACTION_ID,)
    assert "promotional cashback not received" in workflow.excluded_scenarios
    assert "merchant Payment API refund" in workflow.excluded_scenarios
    assert "Google Play app-purchase refund" in workflow.excluded_scenarios
    assert all(url.startswith("https://www.momo.vn/") for url in workflow.source_urls)


def test_wrong_details_and_overdue_cases_are_not_auto_resolved() -> None:
    workflow = BANK_TRANSFER_NOT_RECEIVED_V1

    assert any("wrong recipient" in condition for condition in workflow.handoff_conditions)
    assert any("public policy window" in condition for condition in workflow.handoff_conditions)
    assert "handoff_for_wrong_details_or_overdue_case" in workflow.terminal_outcomes


def test_cashback_workflow_is_separate_and_source_backed() -> None:
    workflow = get_workflow(WorkflowName.CASHBACK_NOT_RECEIVED_V1)

    assert workflow is CASHBACK_NOT_RECEIVED_V1
    assert workflow.required_slots == (SlotName.TRANSACTION_ID,)
    assert "cashback still within the public 24-hour window" in workflow.included_scenarios
    assert "merchant Payment API refund" in workflow.excluded_scenarios
    assert all(url.startswith("https://www.momo.vn/") for url in workflow.source_urls)
