import json
from pathlib import Path

from customer_ops_agent.qa import run_answer_qa
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
