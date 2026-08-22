import json
from pathlib import Path

from customer_ops_agent.qa import evaluate_review_gate, run_answer_qa
from customer_ops_agent.agent_harness import RuleBasedAgent


ROOT = Path(__file__).parents[1]


def test_answer_qa_produces_pending_human_review_records(tmp_path: Path) -> None:
    output = tmp_path / "qa.json"
    report = run_answer_qa(
        ROOT / "data/golden/bank_transfer_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        RuleBasedAgent(),
    )
    output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    assert report.total == 11
    assert report.automated_passed == 11
    assert all(record.review_status == "pending" for record in report.records)
    assert all(record.response for record in report.records)
    assert report.records[0].response_handle == "ask_for_transaction_id"
    assert report.records[0].response != report.records[0].response_handle
    assert report.records[1].response_handle == "answer"
    assert output.exists()


def test_answer_qa_preserves_human_review_status(tmp_path: Path) -> None:
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": "bank-transfer-v1-001",
                        "customer_message": "Tôi chuyển khoản ngân hàng từ MoMo nhưng người nhận chưa nhận được tiền.",
                        "response": "Bạn vui lòng gửi mã giao dịch để mình kiểm tra tình trạng chuyển khoản cho bạn.",
                        "actual_outcome": "ask_for_transaction_id",
                        "policy_message_key": None,
                        "source_document_id": None,
                        "review_status": "approved",
                        "reviewer_notes": "Looks good.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_answer_qa(
        ROOT / "data/golden/bank_transfer_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        RuleBasedAgent(),
        previous_path=previous,
    )

    first = report.records[0]
    assert first.review_status == "approved"
    assert first.reviewer_notes == "Looks good."


def test_answer_qa_resets_approval_when_reviewed_response_changes(tmp_path: Path) -> None:
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": "cashback-v1-002",
                        "customer_message": "Khoản hoàn cashback của giao dịch txn_demo_201 chưa về, mới phát sinh hôm nay.",
                        "response": "Một câu trả lời cũ đã được duyệt.",
                        "actual_outcome": "cashback_pending_within_24_hours",
                        "policy_message_key": "cashback_pending_within_24_hours",
                        "source_document_id": "official-faq-cashback-not-received-2026-08-22",
                        "review_status": "approved",
                        "reviewer_notes": "Looks good.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_answer_qa(
        ROOT / "data/golden/cashback_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        RuleBasedAgent(),
        previous_path=previous,
    )

    changed = next(record for record in report.records if record.case_id == "cashback-v1-002")
    assert changed.review_status == "pending"
    assert changed.reviewer_notes == ""


def test_review_gate_blocks_pending_cases() -> None:
    report = run_answer_qa(
        ROOT / "data/golden/cashback_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        RuleBasedAgent(),
    )

    gate = evaluate_review_gate(report)

    assert gate.passed is False
    assert gate.total == 8
    assert gate.automated_passed == 8
    assert gate.approved == 0
    assert gate.pending == 8
    assert gate.rejected == 0


def test_review_gate_requires_all_cases_approved() -> None:
    report = run_answer_qa(
        ROOT / "data/golden/cashback_not_received_v1.jsonl",
        ROOT / "data/golden/fixtures.json",
        RuleBasedAgent(),
    )
    approved = report.model_copy(
        update={
            "records": tuple(
                record.model_copy(
                    update={"review_status": "approved", "reviewer_notes": "Reviewed."}
                )
                for record in report.records
            )
        }
    )

    gate = evaluate_review_gate(approved)

    assert gate.passed is True
    assert gate.approved == 8
    assert gate.pending == 0
