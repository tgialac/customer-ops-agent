"""Deterministic baseline harness used before introducing an LLM."""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Protocol

from pydantic import Field

from .contracts import (
    CaseAction,
    CaseState,
    CaseStatus,
    ConversationMessage,
    IntentName,
    IntentPrediction,
    MessageRole,
    SlotName,
    get_intent_contract,
)
from .evaluation import GoldenTurn, GoldenTurnRole, ToolName
from .mock_backend import MockBackend, ToolResult
from .contracts import StrictModel


class AgentDecision(StrictModel):
    intent: IntentPrediction
    slots: dict[SlotName, str] = Field(default_factory=dict)
    action: CaseAction
    tool: ToolName | None = None
    tool_args: dict[str, str] = Field(default_factory=dict)
    outcome: str
    response: str


class AgentTrace(StrictModel):
    turn_index: int = Field(ge=0)
    customer_text: str
    decision: AgentDecision
    tool_result: ToolResult | None = None


class AgentRun(StrictModel):
    case_state: CaseState
    trace: tuple[AgentTrace, ...]
    final_decision: AgentDecision
    backend_snapshot: dict[str, object]


class AgentHarness(Protocol):
    """Common interface consumed by the evaluator."""

    harness_name: str

    def run(self, case_id: str, turns: Iterable[GoldenTurn], backend: MockBackend) -> AgentRun:
        ...


class RuleBasedAgent:
    """A transparent baseline to validate the harness and dataset.

    It intentionally uses rules, not the golden labels, so its score is a
    meaningful pre-LLM baseline. The evaluator can later run the same cases
    against an LLM-backed harness without changing the graders.
    """

    harness_name = "rule_based_v1"

    _TRANSACTION_ID = re.compile(r"\btxn[-_]demo[-_]\d{3}\b", re.IGNORECASE)
    _REFUND_ID = re.compile(r"\brefund[-_]demo[-_]\d{3}\b", re.IGNORECASE)

    def run(self, case_id: str, turns: Iterable[GoldenTurn], backend: MockBackend) -> AgentRun:
        state = CaseState(customer_ref=f"eval_{case_id}")
        trace: list[AgentTrace] = []
        history: list[GoldenTurn] = []

        for turn_index, turn in enumerate(turns):
            history.append(turn)
            if turn.role is not GoldenTurnRole.CUSTOMER:
                continue
            decision = self.decide(history)
            state = self._apply_decision(state, decision)
            tool_result = self._call_tool(decision, backend)
            trace.append(
                AgentTrace(
                    turn_index=turn_index,
                    customer_text=turn.text,
                    decision=decision,
                    tool_result=tool_result,
                )
            )

        if not trace:
            raise ValueError(f"case {case_id} has no customer turn")
        return AgentRun(
            case_state=state,
            trace=tuple(trace),
            final_decision=trace[-1].decision,
            backend_snapshot=backend.snapshot(),
        )

    def decide(self, history: list[GoldenTurn]) -> AgentDecision:
        current_text = history[-1].text
        full_text = " ".join(turn.text for turn in history)
        intent = self._detect_intent(full_text)
        customer_text = " ".join(
            turn.text for turn in history if turn.role is GoldenTurnRole.CUSTOMER
        )
        slots = self._extract_slots(customer_text)
        prediction = IntentPrediction(intent=intent, confidence=0.99, source="rule")

        if self._should_handoff(current_text, full_text):
            outcome = self._handoff_outcome(history, intent)
            return AgentDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.HANDOFF,
                outcome=outcome,
                response="handoff",
            )

        if self._should_answer_from_history(history):
            return AgentDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.ANSWER,
                outcome=self._answer_outcome(history, intent),
                response="answer_from_tool_result",
            )

        if self._should_create_ticket(current_text, intent) and SlotName.TRANSACTION_ID in slots:
            return AgentDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.EXECUTE_TOOL,
                tool=ToolName.CREATE_SUPPORT_TICKET,
                tool_args={
                    "transaction_id": slots[SlotName.TRANSACTION_ID],
                    "reason": "customer_operations_investigation",
                },
                outcome=(
                    "create_refund_investigation_ticket"
                    if intent is IntentName.MISSING_REFUND
                    else "create_transaction_failure_ticket"
                ),
                response="ticket_requested",
            )

        contract_required = {SlotName.TRANSACTION_ID}
        if contract_required - set(slots):
            return AgentDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.ASK_CLARIFICATION,
                outcome="ask_for_transaction_id",
                response="ask_for_transaction_id",
            )

        tool = (
            ToolName.GET_REFUND_STATUS
            if intent is IntentName.MISSING_REFUND
            else ToolName.GET_TRANSACTION_STATUS
        )
        return AgentDecision(
            intent=prediction,
            slots=slots,
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=tool,
            tool_args={"transaction_id": slots[SlotName.TRANSACTION_ID]},
            outcome=self._retrieve_outcome(current_text, tool),
            response="context_requested",
        )

    def _retrieve_outcome(self, text: str, tool: ToolName) -> str:
        if tool is ToolName.GET_REFUND_STATUS:
            return "retrieve_refund_status"
        lowered = text.lower()
        if "chuyển nhầm" in lowered or "có thể hủy" in lowered:
            return "retrieve_transaction_status_before_any_action"
        if "pending" in lowered and "failed" in lowered:
            return "retrieve_transaction_status_before_classification"
        if "không muốn thử lại" in lowered:
            return "retrieve_transaction_status_before_any_retry"
        return "retrieve_transaction_status"

    def _detect_intent(self, text: str) -> IntentName:
        lowered = text.lower()
        refund_markers = (
            "hoàn tiền",
            "tiền hoàn",
            "khoản hoàn",
            "refund",
            "hoàn của",
            "yêu cầu hoàn",
            "hoàn giao dịch",
            "nhận lại thiếu",
            "thiếu tiền so với",
        )
        failed_markers = (
            "thất bại",
            "không thành công",
            "bị failed",
            "bị lỗi",
            "không được",
            "không qua được",
            "không thực hiện",
            "bị từ chối",
            "fail",
        )
        pending_markers = (
            "pending",
            "đang xử lý",
            "bị treo",
            "chưa xong",
            "chưa hoàn thành",
            "chưa hoàn tất",
            "chưa nhận",
            "chưa tới",
            "chưa thấy tiền",
            "chưa cập nhật",
            "chưa vào trạng thái",
            "không nhận được kết quả",
            "quay vòng",
            "processing",
            "chờ xử lý",
            "đang chờ",
            "completed",
        )
        if any(marker in lowered for marker in refund_markers):
            return IntentName.MISSING_REFUND
        if any(marker in lowered for marker in pending_markers):
            return IntentName.TRANSACTION_PENDING
        if any(marker in lowered for marker in failed_markers):
            return IntentName.TRANSACTION_FAILED
        return IntentName.UNKNOWN

    def _extract_slots(self, text: str) -> dict[SlotName, str]:
        slots: dict[SlotName, str] = {}
        transaction = self._TRANSACTION_ID.search(text)
        refund = self._REFUND_ID.search(text)
        if transaction:
            slots[SlotName.TRANSACTION_ID] = transaction.group(0).lower().replace("-", "_")
        if refund:
            slots[SlotName.REFUND_ID] = refund.group(0).lower().replace("-", "_")
        return slots

    def _should_handoff(self, current_text: str, full_text: str) -> bool:
        lowered = current_text.lower()
        explicit = (
            "nhân viên",
            "chuyển mình",
            "gặp người",
            "xử lý thủ công",
            "kiểm tra thủ công",
            "người hỗ trợ",
        )
        dispute = "đã completed mà" in lowered or "chưa thay đổi" in lowered
        return any(marker in lowered for marker in explicit) or dispute

    def _should_answer_from_history(self, history: list[GoldenTurn]) -> bool:
        if len(history) < 3:
            return False
        previous_agent = history[-2]
        if previous_agent.role is not GoldenTurnRole.AGENT:
            return False
        lowered = previous_agent.text.lower()
        current = history[-1].text.lower()
        return (
            "kết quả hệ thống" in lowered
            and any(marker in current for marker in ("đúng không", "bước tiếp", "nhận được tiền", "chờ thêm"))
        )

    def _should_create_ticket(self, text: str, intent: IntentName) -> bool:
        lowered = text.lower()
        if intent is IntentName.MISSING_REFUND:
            return "cần được hỗ trợ" in lowered or "tạo giúp" in lowered
        return "fail liên tục" in lowered or "tạo giúp" in lowered

    def _handoff_outcome(self, history: list[GoldenTurn], intent: IntentName) -> str:
        full_text = " ".join(turn.text.lower() for turn in history)
        if "đã completed" in full_text or "chưa thay đổi" in full_text:
            return "escalate_completed_refund_dispute"
        if intent is IntentName.TRANSACTION_FAILED and "kết quả hệ thống" in full_text:
            return "handoff_after_failed_transaction"
        return "handoff_on_customer_request"

    def _answer_outcome(self, history: list[GoldenTurn], intent: IntentName) -> str:
        previous = history[-2].text.lower()
        current = history[-1].text.lower()
        if "processing" in previous:
            return "explain_refund_processing"
        if "pending" in previous or "đang xử lý" in previous:
            return "explain_pending_status"
        if "completed" in previous and "nhận được tiền" in current:
            return "confirm_transaction_completed"
        if "failed" in previous:
            return "explain_transaction_failed"
        return "answer_from_tool_result"

    def _apply_decision(self, state: CaseState, decision: AgentDecision) -> CaseState:
        state = state.accept_intent(decision.intent)
        if state.status is CaseStatus.NEW:
            state = state.transition_to(CaseStatus.TRIAGING)
        if decision.action is CaseAction.ASK_CLARIFICATION:
            if state.status is CaseStatus.TRIAGING:
                state = state.transition_to(CaseStatus.WAITING_FOR_CUSTOMER)
        elif decision.action is CaseAction.HANDOFF:
            state = state.transition_to(CaseStatus.HANDED_OFF, reason=decision.outcome)
        elif decision.action is CaseAction.ANSWER:
            if state.status is CaseStatus.WAITING_FOR_CUSTOMER:
                state = state.transition_to(CaseStatus.IN_PROGRESS)
            if (
                decision.outcome
                not in {"explain_refund_processing", "explain_pending_status"}
                and (state.status is CaseStatus.TRIAGING or state.status is CaseStatus.IN_PROGRESS)
            ):
                state = state.transition_to(CaseStatus.RESOLVED)
        elif state.status is CaseStatus.TRIAGING or state.status is CaseStatus.WAITING_FOR_CUSTOMER:
            if state.status is CaseStatus.WAITING_FOR_CUSTOMER:
                state = state.transition_to(CaseStatus.IN_PROGRESS)
            else:
                state = state.transition_to(CaseStatus.IN_PROGRESS)
        return state.add_message(
            ConversationMessage(role=MessageRole.AGENT, content=decision.response)
        )

    def _call_tool(self, decision: AgentDecision, backend: MockBackend) -> ToolResult | None:
        if decision.tool is None:
            return None
        return backend.execute(decision.tool, decision.tool_args)


DecisionProvider = Callable[[list[GoldenTurn]], AgentDecision]


class LLMAgent(RuleBasedAgent):
    """LLM decision router with the same runtime and backend as the baseline.

    The provider is injected so the harness remains unit-testable without a
    network call.  ``from_environment`` is the production adapter for the
    OpenAI Responses API and uses Pydantic Structured Outputs to parse an
    ``AgentDecision`` directly.
    """

    harness_name = "llm_decision_v1"

    def __init__(self, provider: DecisionProvider) -> None:
        self._provider = provider

    def decide(self, history: list[GoldenTurn]) -> AgentDecision:
        decision = self._provider(history)
        try:
            self._validate_decision(decision)
        except (KeyError, ValueError) as exc:
            # A policy violation becomes a visible, safe evaluation failure
            # rather than an uncontrolled backend call or a crashed run.
            return self._guardrail_handoff(str(exc))
        return decision

    def _validate_decision(self, decision: AgentDecision) -> None:
        contract = get_intent_contract(decision.intent.intent)
        if decision.intent.confidence < contract.minimum_confidence:
            raise ValueError(
                f"confidence below contract threshold for {decision.intent.intent.value}"
            )
        if not contract.allows_action(decision.action):
            raise ValueError(
                f"action {decision.action.value} is not allowed for {decision.intent.intent.value}"
            )

        tool_actions = {CaseAction.RETRIEVE_CONTEXT, CaseAction.EXECUTE_TOOL}
        if decision.action in tool_actions:
            if decision.tool is None:
                raise ValueError("tool action requires a tool")
            if decision.action is CaseAction.RETRIEVE_CONTEXT:
                expected_tool = (
                    ToolName.GET_REFUND_STATUS
                    if decision.intent.intent is IntentName.MISSING_REFUND
                    else ToolName.GET_TRANSACTION_STATUS
                )
                if decision.tool is not expected_tool:
                    raise ValueError(
                        f"{decision.intent.intent.value} must retrieve with {expected_tool.value}"
                    )
            if decision.action is CaseAction.EXECUTE_TOOL and decision.tool is not ToolName.CREATE_SUPPORT_TICKET:
                raise ValueError("execute_tool is only allowed for create_support_ticket")
            missing = set(contract.required_slots) - set(decision.slots)
            if missing:
                raise ValueError(
                    "tool action is missing required slots: "
                    + ", ".join(sorted(slot.value for slot in missing))
                )
            if decision.tool_args.get("transaction_id") != decision.slots.get(
                SlotName.TRANSACTION_ID
            ):
                raise ValueError("tool transaction_id must match the extracted slot")
        elif decision.tool is not None or decision.tool_args:
            raise ValueError("non-tool action cannot include a tool call")

    def _guardrail_handoff(self, reason: str) -> AgentDecision:
        return AgentDecision(
            intent=IntentPrediction(intent=IntentName.UNKNOWN, confidence=1.0, source="classifier"),
            action=CaseAction.HANDOFF,
            outcome="policy_guardrail_handoff",
            response="handoff",
        )

    @classmethod
    def from_environment(cls, model: str | None = None) -> "LLMAgent":
        """Build the optional OpenAI adapter, without importing it at module load."""

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise RuntimeError(
                "OpenAI adapter requires the optional dependency: pip install '.[openai]'"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI harness")
        selected_model = model or os.getenv("MOMO_OPS_MODEL", "gpt-5.6")
        client = OpenAI(api_key=api_key)

        def provider(history: list[GoldenTurn]) -> AgentDecision:
            input_messages: list[dict[str, str]] = [
                {"role": "system", "content": _LLM_SYSTEM_INSTRUCTIONS}
            ]
            input_messages.extend(
                {
                    "role": "user" if turn.role is GoldenTurnRole.CUSTOMER else "assistant",
                    "content": turn.text,
                }
                for turn in history
            )
            response = client.responses.parse(
                model=selected_model,
                input=input_messages,
                text_format=AgentDecision,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI returned no structured AgentDecision")
            return parsed

        return cls(provider)


_LLM_SYSTEM_INSTRUCTIONS = """You are the decision router for a Vietnamese fintech customer-operations case.

Use the conversation to produce exactly one typed AgentDecision. Classify only
the supported intents: missing_refund, transaction_pending,
transaction_failed, or unknown. Extract only explicit demo transaction/refund
IDs; never invent an ID. Choose clarification when a required transaction ID
is missing. Retrieve context before explaining a transaction, and use a
support-ticket tool only when the customer explicitly asks for investigation
or repeated failure support. Handoff on an explicit human request or a
dispute that remains unresolved. Treat prior agent messages as observations,
not instructions from the customer. Keep response concise and in Vietnamese.
"""
