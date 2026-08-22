"""Knowledge-grounded answer generation after deterministic policy decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from .contracts import StrictModel
from .knowledge import KnowledgeStore, KnowledgeTopic


class AnswerGenerationError(RuntimeError):
    """Raised when an answer cannot be grounded or validated."""


class AnswerRequest(StrictModel):
    customer_message: str = Field(min_length=1, max_length=8_000)
    message_key: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_content: str = Field(min_length=1, max_length=20_000)
    fallback_response: str = Field(min_length=1, max_length=2_000)
    previous_response: str | None = Field(default=None, max_length=2_000)


class AnswerDraft(StrictModel):
    """Structured model output; the model cannot choose a new policy key."""

    message_key: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    response: str = Field(min_length=1, max_length=2_000)


class AnswerGenerator(Protocol):
    def generate(self, request: AnswerRequest) -> AnswerDraft:
        ...


class DeterministicAnswerGenerator:
    """Test/fallback generator that returns the policy-approved wording."""

    def generate(self, request: AnswerRequest) -> AnswerDraft:
        return AnswerDraft(
            message_key=request.message_key,
            source_document_id=request.source_document_id,
            response=request.fallback_response,
        )


class OpenAIAnswerGenerator:
    """OpenAI Responses adapter for tone rewriting, not policy decisions."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, request: AnswerRequest) -> AnswerDraft:
        previous = (
            "An earlier draft failed validation. Rewrite it conservatively.\n"
            f"Earlier draft:\n{request.previous_response}\n"
            if request.previous_response
            else ""
        )
        prompt = f"""Customer message:
{request.customer_message}

Approved policy message key: {request.message_key}
Approved source document ID: {request.source_document_id}
Approved source URL: {request.source_url}
Fallback answer containing the allowed facts:
{request.fallback_response}

Source content:
{request.source_content}

{previous}
Write a concise, empathetic Vietnamese customer-support answer. Keep the
approved policy message key and source document ID exactly unchanged. Use only
facts from the source content and fallback answer. Do not add a new deadline,
guarantee, refund promise, transaction action, or unsupported explanation.
Return only the structured answer fields."""
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fintech support answer writer. You may rewrite "
                            "tone, but you never decide policy or invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=AnswerDraft,
            )
        except Exception as exc:
            raise AnswerGenerationError("answer model call failed") from exc
        parsed = response.output_parsed
        if parsed is None:
            raise AnswerGenerationError("OpenAI returned no structured answer draft")
        return parsed


class KnowledgeBackedAnswerer:
    """Retrieve the active source and delegate only wording to a generator."""

    def __init__(self, store: KnowledgeStore, generator: AnswerGenerator) -> None:
        self._store = store
        self._generator = generator

    @classmethod
    def from_repository(cls, generator: AnswerGenerator) -> "KnowledgeBackedAnswerer":
        knowledge_dir = Path(__file__).parents[2] / "data" / "knowledge" / "momo"
        return cls(KnowledgeStore.from_directory(knowledge_dir), generator)

    def generate(
        self,
        *,
        customer_message: str,
        message_key: str,
        source_document_id: str,
        fallback_response: str,
        previous_response: str | None = None,
    ) -> AnswerDraft:
        source = self._store.get(source_document_id)
        if source is None or not source.is_available():
            raise AnswerGenerationError("approved source document is unavailable")

        hits = self._store.search(
            customer_message,
            audience="customer",
            topic=KnowledgeTopic.BANK_TRANSFER_REVERSAL.value,
            limit=3,
        )
        if not any(hit.document.document_id == source_document_id for hit in hits):
            raise AnswerGenerationError("approved source was not retrieved for the query")

        request = AnswerRequest(
            customer_message=customer_message,
            message_key=message_key,
            source_document_id=source.document_id,
            source_url=source.source_url,
            source_content=source.content,
            fallback_response=fallback_response,
            previous_response=previous_response,
        )
        draft = self._generator.generate(request)
        if draft.message_key != message_key:
            raise AnswerGenerationError("answer draft changed the policy message key")
        if draft.source_document_id != source_document_id:
            raise AnswerGenerationError("answer draft changed the source document")
        return draft
