from pathlib import Path

from momo_ops_agent.agent_harness import LLMAgent, RouterDecision, RuleBasedAgent
from momo_ops_agent.answering import DeterministicAnswerGenerator, KnowledgeBackedAnswerer
from momo_ops_agent.contracts import CaseAction, CaseStatus, IntentName, IntentPrediction
from momo_ops_agent.evaluation import GoldenTurn, GoldenTurnRole, OutcomeName
from momo_ops_agent.mock_backend import MockBackend
from momo_ops_agent.policies import GOOGLE_PLAY_REFUND_POLICY_SOURCE


ROOT = Path(__file__).parents[1]


def _turn(text: str) -> list[GoldenTurn]:
    return [GoldenTurn(role=GoldenTurnRole.CUSTOMER, text=text)]


def test_google_play_workflow_answers_without_transaction_id() -> None:
    run = RuleBasedAgent().run(
        "google-play-runtime-001",
        _turn("Tôi muốn hoàn tiền cho ứng dụng đã mua trên Google Play."),
        MockBackend(),
    )

    assert run.final_decision.intent.intent is IntentName.MISSING_REFUND
    assert run.final_decision.action is CaseAction.ANSWER
    assert run.final_decision.outcome is OutcomeName.GOOGLE_PLAY_REFUND_INSTRUCTIONS
    assert run.final_decision.policy_source == GOOGLE_PLAY_REFUND_POLICY_SOURCE
    assert run.final_decision.customer_response is not None
    assert "Báo cáo sự cố" in run.final_decision.customer_response
    assert run.case_state.status is CaseStatus.RESOLVED


def test_google_play_answer_uses_source_and_guardrail() -> None:
    answerer = KnowledgeBackedAnswerer.from_repository(DeterministicAnswerGenerator())
    run = RuleBasedAgent(answerer=answerer).run(
        "google-play-runtime-002",
        _turn("Trong Google Play, tôi phải vào đâu để xin hoàn tiền ứng dụng?"),
        MockBackend(),
    )

    assert run.trace[0].answer_generation_attempts == 1
    assert run.trace[0].output_guardrail is not None
    assert run.trace[0].output_guardrail.passed is True
    assert "Lịch sử đơn đặt hàng" in run.final_decision.response


def test_google_play_result_location_uses_narrow_policy_message() -> None:
    run = RuleBasedAgent().run(
        "google-play-runtime-002b",
        _turn("Tôi đã mua app trên Google Play, kết quả hoàn tiền sẽ được gửi ở đâu?"),
        MockBackend(),
    )

    assert run.final_decision.policy_message_key == "google_play_refund_result_location"
    assert "email" in run.final_decision.response
    assert "Lịch sử đơn đặt hàng" not in run.final_decision.response


def test_llm_router_cannot_turn_google_play_help_into_tool_lookup() -> None:
    def provider(_: list[GoldenTurn]) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(
                intent=IntentName.MISSING_REFUND,
                confidence=0.99,
                source="classifier",
            ),
            action=CaseAction.RETRIEVE_CONTEXT,
        )

    run = LLMAgent(provider).run(
        "google-play-runtime-003",
        _turn("Tôi muốn refund ứng dụng trên Google Play."),
        MockBackend(),
    )

    assert run.final_decision.action is CaseAction.ANSWER
    assert run.trace[0].tool_result is None
    assert run.final_decision.outcome is OutcomeName.GOOGLE_PLAY_REFUND_INSTRUCTIONS
