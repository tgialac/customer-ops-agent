from scripts.prepare_banking77 import normalize_rows


def test_normalize_rows_maps_only_the_initial_project_intents() -> None:
    rows = [
        {"text": "Where is my refund?", "category": "Refund_not_showing_up"},
        {"text": "My transfer is pending.", "category": "pending_transfer"},
        {"text": "The transfer failed.", "category": "failed_transfer"},
        {"text": "How do I change my PIN?", "category": "change_pin"},
    ]

    normalized = normalize_rows(rows, "test")

    assert [row["project_intent"] for row in normalized] == [
        "missing_refund",
        "transaction_pending",
        "transaction_failed",
    ]
    assert [row["id"] for row in normalized] == [
        "banking77-test-00000",
        "banking77-test-00001",
        "banking77-test-00002",
    ]


def test_normalize_rows_rejects_empty_selected_utterances() -> None:
    try:
        normalize_rows(
            [{"text": "  ", "category": "pending_transfer"}],
            "train",
        )
    except ValueError as error:
        assert "empty utterance" in str(error)
    else:
        raise AssertionError("expected empty selected utterance to be rejected")
