from datetime import date
from pathlib import Path

from momo_ops_agent.knowledge import KnowledgeStore, KnowledgeTopic


ROOT = Path(__file__).parents[1]
KNOWLEDGE_DIR = ROOT / "data/knowledge/momo"


def test_public_momo_corpus_has_source_and_excludes_future_terms() -> None:
    store = KnowledgeStore.from_directory(KNOWLEDGE_DIR)

    hits = store.search(
        "giao dịch ngân hàng pending bị trừ tiền chưa nhận được hoàn tiền",
        as_of=date(2026, 8, 22),
    )

    assert hits
    assert hits[0].document.topic == KnowledgeTopic.BANK_TRANSFER_REVERSAL
    assert all(hit.document.source_url.startswith("https://www.momo.vn/") for hit in hits)
    assert all(hit.document.status == "active" for hit in hits)
    assert all(hit.document.effective_from is None for hit in hits)


def test_cashback_policy_is_separate_from_bank_transfer_policy() -> None:
    store = KnowledgeStore.from_directory(KNOWLEDGE_DIR)

    hits = store.search(
        "cashback không được hoàn tiền chờ 24 giờ",
        topic=KnowledgeTopic.CASHBACK_NOT_RECEIVED,
    )

    assert hits[0].document.topic == KnowledgeTopic.CASHBACK_NOT_RECEIVED
    assert all(hit.document.topic != KnowledgeTopic.BANK_TRANSFER_REVERSAL for hit in hits)


def test_customer_search_does_not_expose_merchant_api_policy() -> None:
    store = KnowledgeStore.from_directory(KNOWLEDGE_DIR)

    customer_hits = store.search("hoàn tiền giao dịch thành công hoàn một phần")
    merchant_hits = store.search(
        "hoàn tiền giao dịch thành công hoàn một phần",
        audience="merchant",
        topic=KnowledgeTopic.MERCHANT_REFUND,
    )

    assert all(hit.document.topic != KnowledgeTopic.MERCHANT_REFUND for hit in customer_hits)
    assert merchant_hits
    assert merchant_hits[0].document.source_kind == "official_developer_docs"


def test_google_play_refund_routes_to_google_play_policy() -> None:
    store = KnowledgeStore.from_directory(KNOWLEDGE_DIR)

    hits = store.search("muốn hoàn tiền ứng dụng đã mua trên Google Play")

    assert hits[0].document.topic == KnowledgeTopic.GOOGLE_PLAY_REFUND
