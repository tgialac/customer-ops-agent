from pathlib import Path

import pytest

from customer_ops_agent.answering import (
    AnswerDraft,
    AnswerGenerationError,
    AnswerRequest,
    KnowledgeBackedAnswerer,
    OpenAIAnswerGenerator,
)
from customer_ops_agent.agent_harness import RuleBasedAgent
from customer_ops_agent.contracts import CaseAction, CaseStatus, TransactionStatus
from customer_ops_agent.evaluation import GoldenTurn, GoldenTurnRole, OutcomeName
from customer_ops_agent.knowledge import KnowledgeStore
from customer_ops_agent.mock_backend import MockBackend, TransactionRecord


ROOT = Path(__file__).parents[1]
SOURCE_ID = "official-faq-bank-transfer-reversal-2026-08-22"


class FixedGenerator:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, request: AnswerRequest) -> AnswerDraft:
        return AnswerDraft(
            message_key=request.message_key,
            source_document_id=request.source_document_id,
            response=self.response,
        )


def _answerer(generator: object) -> KnowledgeBackedAnswerer:
    return KnowledgeBackedAnswerer(
        KnowledgeStore.from_directory(ROOT / "data/knowledge/policies"), generator
    )  # type: ignore[arg-type]


def _customer_turn(transaction_id: str) -> list[GoldenTurn]:
    return [
        GoldenTurn(
            role=GoldenTurnRole.CUSTOMER,
            text=(
                "Chuyển khoản ngân hàng, người nhận chưa thấy tiền, "
                f"mã {transaction_id}."
            ),
        )
    ]


def test_knowledge_backed_answerer_requires_retrieved_active_source() -> None:
    answerer = _answerer(FixedGenerator("fallback"))

    draft = answerer.generate(
        customer_message="Chuyển khoản ngân hàng, người nhận chưa thấy tiền.",
        message_key="pending_1_to_2_working_days",
        source_document_id=SOURCE_ID,
        fallback_response="fallback",
    )

    assert draft.source_document_id == SOURCE_ID
    assert draft.message_key == "pending_1_to_2_working_days"

    with pytest.raises(AnswerGenerationError, match="unavailable"):
        answerer.generate(
            customer_message="Chuyển khoản ngân hàng, người nhận chưa thấy tiền.",
            message_key="pending_1_to_2_working_days",
            source_document_id="future-or-unknown-source",
            fallback_response="fallback",
        )


def test_answer_layer_rewrites_tone_but_keeps_policy_facts() -> None:
    answerer = _answerer(
        FixedGenerator(
            "Mình đã kiểm tra: giao dịch đang được đối soát. Bạn vui lòng chờ "
            "1-2 ngày làm việc; nếu sau đó chưa có kết quả, bộ phận hỗ trợ sẽ kiểm tra thêm."
        )
    )
    run = RuleBasedAgent(answerer=answerer).run(
        "answer-layer-001",
        _customer_turn("txn_demo_090"),
        MockBackend(
            [
                TransactionRecord(
                    transaction_id="txn_demo_090",
                    status=TransactionStatus.PENDING,
                    elapsed_working_days=1,
                )
            ]
        ),
    )

    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.outcome is OutcomeName.BANK_TRANSFER_PENDING_RECONCILIATION
    assert "1-2 ngày làm việc" in run.final_decision.response
    assert run.final_decision.customer_response == run.final_decision.response
    assert run.trace[0].answer_generation_attempts == 1
    assert run.trace[0].output_guardrail is not None
    assert run.trace[0].output_guardrail.passed is True
    assert run.case_state.status is CaseStatus.IN_PROGRESS


def test_clarification_and_handoff_have_customer_facing_messages() -> None:
    missing_id = RuleBasedAgent().run(
        "customer-copy-001",
        [
            GoldenTurn(
                role=GoldenTurnRole.CUSTOMER,
                text="Chuyển khoản ngân hàng, người nhận chưa thấy tiền.",
            )
        ],
        MockBackend(),
    )
    assert missing_id.final_decision.response == "ask_for_transaction_id"
    assert missing_id.final_decision.customer_response
    assert missing_id.case_state.messages[-1].content == (
        missing_id.final_decision.customer_response
    )

    handoff = RuleBasedAgent().run(
        "customer-copy-002",
        _customer_turn("txn_demo_092"),
        MockBackend(),
    )
    assert handoff.final_decision.response == "handoff"
    assert handoff.final_decision.customer_response
    assert handoff.case_state.messages[-1].content == (
        handoff.final_decision.customer_response
    )


class RetryGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: AnswerRequest) -> AnswerDraft:
        self.calls += 1
        response = (
            "Giao dịch đang được xử lý."
            if self.calls == 1
            else "Giao dịch đang được đối soát. Vui lòng chờ 1-2 ngày làm việc; "
            "nếu sau đó chưa có kết quả, bộ phận hỗ trợ sẽ kiểm tra thêm."
        )
        return AnswerDraft(
            message_key=request.message_key,
            source_document_id=request.source_document_id,
            response=response,
        )


def test_invalid_generated_answer_is_retried_once_then_accepted() -> None:
    generator = RetryGenerator()
    run = RuleBasedAgent(answerer=_answerer(generator)).run(
        "answer-layer-retry-001",
        _customer_turn("txn_demo_091"),
        MockBackend(
            [
                TransactionRecord(
                    transaction_id="txn_demo_091",
                    status=TransactionStatus.PENDING,
                    elapsed_working_days=1,
                )
            ]
        ),
    )

    assert generator.calls == 2
    assert run.final_decision.action is CaseAction.ANSWER
    assert run.trace[0].answer_generation_attempts == 2
    assert run.trace[0].output_guardrail is not None
    assert run.trace[0].output_guardrail.passed is True


def test_answer_model_api_failure_is_converted_to_generation_error() -> None:
    class FailingResponses:
        def parse(self, **_: object) -> None:
            raise RuntimeError("network failure")

    class FailingClient:
        responses = FailingResponses()

    with pytest.raises(AnswerGenerationError, match="model call failed"):
        OpenAIAnswerGenerator(FailingClient(), "test-model").generate(
            AnswerRequest(
                customer_message="Chuyển khoản ngân hàng chưa nhận được tiền.",
                message_key="pending_1_to_2_working_days",
                source_document_id=SOURCE_ID,
                source_url="https://www.momo.vn/faq",
                source_content="Chờ 1–2 ngày làm việc.",
                fallback_response="Chờ 1–2 ngày làm việc.",
            )
        )
