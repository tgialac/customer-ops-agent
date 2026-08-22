"""Source-backed knowledge retrieval for customer-operations answers.

The first implementation is deliberately local and lexical.  It provides a
stable interface for a future vector store without treating every document as
valid for every audience or product flow.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

from .contracts import StrictModel


class KnowledgeTopic(str, Enum):
    """Topics covered by the public MoMo policy corpus."""

    BANK_TRANSFER_REVERSAL = "bank_transfer_reversal"
    CASHBACK_NOT_RECEIVED = "cashback_not_received"
    MERCHANT_REFUND = "merchant_refund"
    GOOGLE_PLAY_REFUND = "google_play_refund"
    GENERAL_TRANSACTION_DISPUTE = "general_transaction_dispute"


class KnowledgeDocument(StrictModel):
    document_id: str
    title: str
    source_url: str
    source_kind: Literal["official_faq", "official_developer_docs", "official_terms"]
    audience: Literal["customer", "merchant"]
    topic: str
    checked_at: date
    status: Literal["active", "future", "archived"] = "active"
    effective_from: date | None = None
    keywords: tuple[str, ...]
    content: str

    def is_available(self, *, as_of: date | None = None) -> bool:
        """Return whether this source can be used for an answer today."""

        current_date = as_of or date.today()
        return (
            self.status == "active"
            and (self.effective_from is None or self.effective_from <= current_date)
        )


class KnowledgeHit(StrictModel):
    document: KnowledgeDocument
    score: int = 0


_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value)}


class KnowledgeStore:
    """Small source-filtered retriever used by the answer layer and tests."""

    def __init__(self, documents: Iterable[KnowledgeDocument]) -> None:
        self._documents = tuple(documents)

    @classmethod
    def from_directory(cls, directory: Path) -> "KnowledgeStore":
        documents = [
            _parse_markdown_document(path)
            for path in sorted(directory.rglob("*.md"))
        ]
        return cls(documents)

    def search(
        self,
        query: str,
        *,
        audience: Literal["customer", "merchant"] = "customer",
        topic: str | None = None,
        as_of: date | None = None,
        limit: int = 3,
    ) -> list[KnowledgeHit]:
        if not query.strip():
            return []

        query_tokens = _tokens(query)
        hits: list[KnowledgeHit] = []
        for document in self._documents:
            if document.audience != audience or not document.is_available(as_of=as_of):
                continue
            if topic is not None and document.topic != topic:
                continue

            searchable = _tokens(
                " ".join((document.title, document.topic, *document.keywords, document.content))
            )
            score = len(query_tokens & searchable)
            if score:
                hits.append(KnowledgeHit(document=document, score=score))

        return sorted(hits, key=lambda hit: (-hit.score, hit.document.document_id))[:limit]


def _parse_markdown_document(path: Path) -> KnowledgeDocument:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"knowledge document is missing front matter: {path}")

    _, raw_metadata, content = text.split("---\n", 2)
    metadata: dict[str, str] = {}
    for line in raw_metadata.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid front matter in {path}: {line}")
        metadata[key.strip()] = value.strip()

    keywords = tuple(
        keyword.strip()
        for keyword in metadata.get("keywords", "").split(",")
        if keyword.strip()
    )
    return KnowledgeDocument(
        document_id=metadata["document_id"],
        title=metadata["title"],
        source_url=metadata["source_url"],
        source_kind=metadata["source_kind"],
        audience=metadata["audience"],
        topic=metadata["topic"],
        checked_at=date.fromisoformat(metadata["checked_at"]),
        status=metadata.get("status", "active"),
        effective_from=(
            date.fromisoformat(metadata["effective_from"])
            if metadata.get("effective_from")
            else None
        ),
        keywords=keywords,
        content=content.strip(),
    )
