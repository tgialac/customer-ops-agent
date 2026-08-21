"""Deterministic baseline harness used before introducing an LLM."""

from __future__ import annotations

import re
from typing import Iterable

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


class RuleBasedAgent:
    """A transparent baseline to validate the harness and dataset.

    It intentionally uses rules, not the golden labels, so its score is a
    meaningful pre-LLM baseline. The evaluator can later run the same cases
    against an LLM-backed harness without changing the graders.
    """

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
        if decision.tool is ToolName.GET_TRANSACTION_STATUS:
            return backend.get_transaction_status(decision.tool_args["transaction_id"])
        if decision.tool is ToolName.GET_REFUND_STATUS:
            return backend.get_refund_status(decision.tool_args["transaction_id"])
        if decision.tool is ToolName.CREATE_SUPPORT_TICKET:
            return backend.create_support_ticket(
                decision.tool_args["transaction_id"], decision.tool_args["reason"]
            )
        return None
