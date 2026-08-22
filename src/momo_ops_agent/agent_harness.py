"""Rule baseline and structured LLM router sharing one runtime harness."""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Protocol

from pydantic import Field

from .contracts import (
    CaseAction,
    CaseState,
    CaseStatus,
    ContextScope,
    ConversationMessage,
    ContextSnapshot,
    FundingSource,
    IntentName,
    IntentPrediction,
    MessageRole,
    SlotName,
    TransactionSnapshot,
    TransactionStatus,
    get_intent_contract,
)
from .evaluation import GoldenTurn, GoldenTurnRole, OutcomeName, ToolName
from .mock_backend import MockBackend, ToolResult
from .contracts import StrictModel
from .policies import (
    BankTransferPolicyInput,
    evaluate_bank_transfer_policy,
    render_bank_transfer_response,
)
from .guardrails import GuardrailResult, check_input, check_output
from .answering import (
    AnswerGenerationError,
    KnowledgeBackedAnswerer,
    OpenAIAnswerGenerator,
)


class RouterDecision(StrictModel):
    intent: IntentPrediction
    slots: dict[SlotName, str] = Field(default_factory=dict)
    action: CaseAction
    tool: ToolName | None = None
    tool_args: dict[str, str] = Field(default_factory=dict)
    policy_violation: str | None = None
    policy_source: str | None = None
    policy_message_key: str | None = None
    policy_return_destination: str | None = None


class AgentDecision(RouterDecision):
    """Runtime decision after deterministic policy materialization."""

    outcome: OutcomeName
    response: str


class LLMDecisionPayload(StrictModel):
    """OpenAI-compatible wire schema with fixed fields instead of enum-keyed maps."""

    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)
    transaction_id: str | None = None
    refund_id: str | None = None
    action: CaseAction
    tool: ToolName | None = None
    tool_reason: str | None = None

    def to_router_decision(self) -> RouterDecision:
        slots: dict[SlotName, str] = {}
        if self.transaction_id is not None:
            slots[SlotName.TRANSACTION_ID] = self.transaction_id
        if self.refund_id is not None:
            slots[SlotName.REFUND_ID] = self.refund_id

        tool_args: dict[str, str] = {}
        if self.tool is not None and self.transaction_id is not None:
            tool_args["transaction_id"] = self.transaction_id
        if self.tool is ToolName.CREATE_SUPPORT_TICKET:
            tool_args["reason"] = self.tool_reason or "customer_operations_investigation"

        return RouterDecision(
            intent=IntentPrediction(
                intent=self.intent,
                confidence=self.confidence,
                source="classifier",
            ),
            slots=slots,
            action=self.action,
            tool=self.tool,
            tool_args=tool_args,
        )


class AgentTrace(StrictModel):
    turn_index: int = Field(ge=0)
    customer_text: str
    decision: AgentDecision
    tool_result: ToolResult | None = None
    input_guardrail: GuardrailResult | None = None
    output_guardrail: GuardrailResult | None = None
    answer_generation_attempts: int = Field(default=0, ge=0)


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

    def __init__(self, answerer: KnowledgeBackedAnswerer | None = None) -> None:
        self._answerer = answerer

    def run(self, case_id: str, turns: Iterable[GoldenTurn], backend: MockBackend) -> AgentRun:
        state = CaseState(customer_ref=f"eval_{case_id}")
        trace: list[AgentTrace] = []
        history: list[GoldenTurn] = []

        for turn_index, turn in enumerate(turns):
            history.append(turn)
            if turn.role is not GoldenTurnRole.CUSTOMER:
                continue
            input_guardrail = check_input(turn.text)
            if input_guardrail.passed:
                router_decision = self.decide(history)
                tool_result = self._call_tool(router_decision, backend)
                decision = self.materialize_decision(router_decision, history, tool_result)
            else:
                router_decision = RouterDecision(
                    intent=IntentPrediction(
                        intent=IntentName.UNKNOWN,
                        confidence=1.0,
                        source="rule",
                    ),
                    action=CaseAction.HANDOFF,
                    policy_violation=(
                        f"input_guardrail:{input_guardrail.reason or 'rejected'}"
                    ),
                )
                tool_result = None
                decision = self.materialize_decision(router_decision, history)
            answer_generation_attempts = 0
            if (
                self._answerer is not None
                and decision.intent.intent is IntentName.BANK_TRANSFER_NOT_RECEIVED
                and decision.action is CaseAction.ANSWER
                and decision.policy_message_key is not None
                and decision.policy_source is not None
            ):
                decision, answer_generation_attempts = self._generate_answer(
                    decision, turn.text
                )
            output_guardrail = check_output(
                intent=decision.intent.intent,
                action=decision.action,
                response=decision.response,
                policy_source=decision.policy_source,
                policy_message_key=decision.policy_message_key,
                return_destination=decision.policy_return_destination,
            )
            if not output_guardrail.passed:
                if (
                    self._answerer is not None
                    and decision.intent.intent is IntentName.BANK_TRANSFER_NOT_RECEIVED
                    and decision.policy_message_key is not None
                    and decision.policy_source is not None
                ):
                    decision, attempts = self._generate_answer(
                        decision, turn.text, previous_response=decision.response
                    )
                    answer_generation_attempts += attempts
                    output_guardrail = check_output(
                        intent=decision.intent.intent,
                        action=decision.action,
                        response=decision.response,
                        policy_source=decision.policy_source,
                        policy_message_key=decision.policy_message_key,
                        return_destination=decision.policy_return_destination,
                    )
                if not output_guardrail.passed:
                    decision = self._output_guardrail_handoff(
                        decision, output_guardrail.reason or "rejected"
                    )
            state = self._apply_decision(
                state,
                decision,
                tool_result,
                history,
                input_guardrail,
                output_guardrail,
            )
            trace.append(
                AgentTrace(
                    turn_index=turn_index,
                    customer_text=turn.text,
                    decision=decision,
                    tool_result=tool_result,
                    input_guardrail=input_guardrail,
                    output_guardrail=output_guardrail,
                    answer_generation_attempts=answer_generation_attempts,
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

    def decide(self, history: list[GoldenTurn]) -> RouterDecision:
        current_text = history[-1].text
        full_text = " ".join(turn.text for turn in history)
        intent = self._detect_intent(full_text)
        customer_text = " ".join(
            turn.text for turn in history if turn.role is GoldenTurnRole.CUSTOMER
        )
        slots = self._extract_slots(customer_text)
        prediction = IntentPrediction(intent=intent, confidence=0.99, source="rule")

        if self._should_handoff(current_text, full_text):
            return RouterDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.HANDOFF,
            )

        if self._should_answer_from_history(history):
            return RouterDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.ANSWER,
            )

        if self._should_create_ticket(current_text, intent) and SlotName.TRANSACTION_ID in slots:
            return RouterDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.EXECUTE_TOOL,
                tool=ToolName.CREATE_SUPPORT_TICKET,
                tool_args={
                    "transaction_id": slots[SlotName.TRANSACTION_ID],
                    "reason": "customer_operations_investigation",
                },
            )

        contract_required = {SlotName.TRANSACTION_ID}
        if contract_required - set(slots):
            return RouterDecision(
                intent=prediction,
                slots=slots,
                action=CaseAction.ASK_CLARIFICATION,
            )

        tool = (
            ToolName.GET_REFUND_STATUS
            if intent is IntentName.MISSING_REFUND
            else ToolName.GET_TRANSACTION_STATUS
        )
        return RouterDecision(
            intent=prediction,
            slots=slots,
            action=CaseAction.RETRIEVE_CONTEXT,
            tool=tool,
            tool_args={"transaction_id": slots[SlotName.TRANSACTION_ID]},
        )

    def materialize_decision(
        self,
        router_decision: RouterDecision,
        history: list[GoldenTurn],
        tool_result: ToolResult | None = None,
    ) -> AgentDecision:
        """Derive benchmark events and response handles outside the model."""

        intent = router_decision.intent.intent
        if (
            intent is IntentName.BANK_TRANSFER_NOT_RECEIVED
            and router_decision.action is CaseAction.RETRIEVE_CONTEXT
        ):
            return self._materialize_bank_transfer_decision(
                router_decision, history, tool_result
            )
        if router_decision.policy_violation is not None:
            outcome = OutcomeName.POLICY_GUARDRAIL_HANDOFF
            response = "handoff"
        elif router_decision.action is CaseAction.ASK_CLARIFICATION:
            outcome = OutcomeName.ASK_FOR_TRANSACTION_ID
            response = "ask_for_transaction_id"
        elif router_decision.action is CaseAction.HANDOFF:
            outcome = OutcomeName(self._handoff_outcome(history, intent))
            response = "handoff"
        elif router_decision.action is CaseAction.ANSWER:
            outcome = OutcomeName(self._answer_outcome(history, intent))
            response = "answer_from_tool_result"
        elif router_decision.action is CaseAction.EXECUTE_TOOL:
            outcome = OutcomeName(
                "create_refund_investigation_ticket"
                if intent is IntentName.MISSING_REFUND
                else "create_transaction_failure_ticket"
            )
            response = "ticket_requested"
        elif router_decision.action is CaseAction.RETRIEVE_CONTEXT:
            if router_decision.tool is None:
                outcome = OutcomeName.POLICY_GUARDRAIL_HANDOFF
            else:
                outcome = OutcomeName(
                    self._retrieve_outcome(history[-1].text, router_decision.tool)
                )
            response = "context_requested"
        else:
            outcome = OutcomeName.CLOSE_CASE
            response = "close_case"

        return AgentDecision(
            **router_decision.model_dump(exclude={"policy_violation"}),
            outcome=outcome,
            response=response,
        )

    def _materialize_bank_transfer_decision(
        self,
        router_decision: RouterDecision,
        history: list[GoldenTurn],
        tool_result: ToolResult | None,
    ) -> AgentDecision:
        base = router_decision.model_dump(exclude={"policy_violation"})
        if tool_result is None or not tool_result.success:
            base.update(
                action=CaseAction.HANDOFF,
                tool=None,
                tool_args={},
                policy_source=None,
            )
            return AgentDecision(
                **base,
                outcome=OutcomeName.BANK_TRANSFER_TOOL_FAILURE_HANDOFF,
                response="handoff",
            )

        data = tool_result.data
        status = _parse_transaction_status(data.get("status"))
        funding_source = _parse_funding_source(data.get("funding_source"))
        policy = evaluate_bank_transfer_policy(
            BankTransferPolicyInput(
                transaction_id=data.get("transaction_id"),
                status=status,
                funding_source=funding_source,
                elapsed_working_days=data.get("elapsed_working_days"),
                return_elapsed_working_days=data.get("return_elapsed_working_days"),
                wrong_details_reported=self._wrong_bank_details_reported(history),
            )
        )
        base.update(
            tool=None,
            tool_args={},
            policy_source=policy.source_document_id,
            policy_message_key=policy.message_key,
            policy_return_destination=policy.return_destination,
        )
        if policy.handoff_required:
            base["action"] = CaseAction.HANDOFF
            return AgentDecision(
                **base,
                outcome=OutcomeName.BANK_TRANSFER_HANDOFF,
                response="handoff",
            )

        outcome_by_message = {
            "pending_1_to_2_working_days": OutcomeName.BANK_TRANSFER_PENDING_RECONCILIATION,
            "successful_transfer_1_to_3_working_days": (
                OutcomeName.BANK_TRANSFER_DELAYED_BENEFICIARY_POSTING
            ),
            "failed_transfer_return_1_to_2_working_days": (
                OutcomeName.BANK_TRANSFER_FAILED_RETURN
            ),
        }
        outcome = outcome_by_message.get(policy.message_key)
        if outcome is None:
            base["action"] = CaseAction.HANDOFF
            return AgentDecision(
                **base,
                outcome=OutcomeName.BANK_TRANSFER_HANDOFF,
                response="handoff",
            )
        base["action"] = CaseAction.ANSWER
        return AgentDecision(
            **base,
            outcome=outcome,
            response=render_bank_transfer_response(policy),
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
        bank_transfer_markers = (
            "chuyển khoản ngân hàng",
            "tài khoản ngân hàng",
            "ngân hàng người nhận",
            "thẻ ngân hàng",
            "bank account",
        )
        recipient_not_received_markers = (
            "người nhận chưa nhận",
            "người nhận chưa thấy",
            "chưa nhận được tiền",
            "chưa thấy tiền",
        )
        if any(marker in lowered for marker in bank_transfer_markers) and any(
            marker in lowered for marker in recipient_not_received_markers
        ):
            return IntentName.BANK_TRANSFER_NOT_RECEIVED
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

    def _apply_decision(
        self,
        state: CaseState,
        decision: AgentDecision,
        tool_result: ToolResult | None = None,
        history: list[GoldenTurn] | None = None,
        input_guardrail: GuardrailResult | None = None,
        output_guardrail: GuardrailResult | None = None,
    ) -> CaseState:
        if input_guardrail is not None and not input_guardrail.passed:
            state = state.record_guardrail_failure("input")
        if output_guardrail is not None and not output_guardrail.passed:
            state = state.record_guardrail_failure("output")
        state = state.accept_intent(decision.intent)
        if (
            tool_result is not None
            and tool_result.success
            and tool_result.tool_name is ToolName.GET_TRANSACTION_STATUS
            and decision.intent.intent is not IntentName.UNKNOWN
            and get_intent_contract(decision.intent.intent).allows_context(
                ContextScope.TRANSACTION
            )
        ):
            state = self._add_transaction_context(
                state, tool_result, self._wrong_bank_details_reported(history or [])
            )
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
            elif state.status is CaseStatus.TRIAGING:
                state = state.transition_to(CaseStatus.IN_PROGRESS)
            if (
                decision.outcome
                not in {
                    "explain_refund_processing",
                    "explain_pending_status",
                    OutcomeName.BANK_TRANSFER_PENDING_RECONCILIATION,
                    OutcomeName.BANK_TRANSFER_DELAYED_BENEFICIARY_POSTING,
                    OutcomeName.BANK_TRANSFER_FAILED_RETURN,
                }
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

    def _call_tool(self, decision: RouterDecision, backend: MockBackend) -> ToolResult | None:
        if decision.tool is None:
            return None
        return backend.execute(decision.tool, decision.tool_args)

    def _output_guardrail_handoff(
        self, decision: AgentDecision, reason: str
    ) -> AgentDecision:
        values = decision.model_dump(
            exclude={
                "action",
                "tool",
                "tool_args",
                "policy_violation",
                "outcome",
                "response",
            }
        )
        return AgentDecision(
            **values,
            action=CaseAction.HANDOFF,
            tool=None,
            tool_args={},
            policy_violation=f"output_guardrail:{reason}",
            outcome=OutcomeName.POLICY_GUARDRAIL_HANDOFF,
            response="handoff",
        )

    def _generate_answer(
        self,
        decision: AgentDecision,
        customer_message: str,
        *,
        previous_response: str | None = None,
    ) -> tuple[AgentDecision, int]:
        if self._answerer is None or decision.policy_message_key is None:
            return decision, 0
        try:
            draft = self._answerer.generate(
                customer_message=customer_message,
                message_key=decision.policy_message_key,
                source_document_id=decision.policy_source or "",
                fallback_response=decision.response,
                previous_response=previous_response,
            )
        except AnswerGenerationError:
            return self._output_guardrail_handoff(decision, "answer_generation_failed"), 1
        return decision.model_copy(update={"response": draft.response}), 1

    def _add_transaction_context(
        self, state: CaseState, tool_result: ToolResult, wrong_details_reported: bool
    ) -> CaseState:
        data = tool_result.data
        try:
            context = ContextSnapshot(
                scopes=(ContextScope.TRANSACTION,),
                transaction=TransactionSnapshot(
                    transaction_id=str(data["transaction_id"]),
                    status=TransactionStatus(str(data["status"])),
                    amount_minor=int(data["amount_minor"]),
                    currency=str(data["currency"]),
                    funding_source=_parse_funding_source(data.get("funding_source")),
                    elapsed_working_days=data.get("elapsed_working_days"),
                    return_elapsed_working_days=data.get("return_elapsed_working_days"),
                    wrong_details_reported=wrong_details_reported,
                ),
            )
        except (KeyError, TypeError, ValueError):
            return state
        return state.with_context(context)

    def _wrong_bank_details_reported(self, history: list[GoldenTurn]) -> bool:
        text = " ".join(turn.text.lower() for turn in history)
        return any(
            marker in text
            for marker in (
                "chuyển nhầm",
                "sai số tài khoản",
                "sai thông tin tài khoản",
                "nhầm tài khoản",
                "nhầm ngân hàng",
            )
        )


def _parse_transaction_status(value: object) -> TransactionStatus | None:
    try:
        return TransactionStatus(str(value)) if value is not None else None
    except ValueError:
        return None


def _parse_funding_source(value: object) -> FundingSource:
    try:
        return FundingSource(str(value)) if value is not None else FundingSource.UNKNOWN
    except ValueError:
        return FundingSource.UNKNOWN


DecisionProvider = Callable[[list[GoldenTurn]], RouterDecision]


class LLMAgent(RuleBasedAgent):
    """LLM decision router with the same runtime and backend as the baseline.

    The provider is injected so the harness remains unit-testable without a
    network call. ``from_environment`` is the production adapter for the
    OpenAI Responses API and uses Pydantic Structured Outputs to parse only a
    ``RouterDecision``. Runtime outcomes and responses are materialized by the
    deterministic policy layer.
    """

    harness_name = "llm_decision_v1"

    def __init__(
        self,
        provider: DecisionProvider,
        answerer: KnowledgeBackedAnswerer | None = None,
    ) -> None:
        super().__init__(answerer=answerer)
        self._provider = provider

    def decide(self, history: list[GoldenTurn]) -> RouterDecision:
        decision = self._normalize_router_decision(self._provider(history), history)
        try:
            self._validate_decision(decision)
        except (KeyError, ValueError) as exc:
            # A policy violation becomes a visible, safe evaluation failure
            # rather than an uncontrolled backend call or a crashed run.
            return self._guardrail_handoff(str(exc))
        return decision

    def _normalize_router_decision(
        self, decision: RouterDecision, history: list[GoldenTurn]
    ) -> RouterDecision:
        """Apply deterministic safety/policy normalization before validation.

        Entity extraction and high-risk routing rules are application-owned:
        the model may propose a route, but it cannot silently invent an ID,
        select an unrelated tool, or turn an explicit ticket request into an
        arbitrary tool call.
        """

        customer_text = " ".join(
            turn.text for turn in history if turn.role is GoldenTurnRole.CUSTOMER
        )
        explicit_slots = self._extract_slots(customer_text)
        # IDs explicitly present in the conversation are canonical. Discard
        # model-only slots so a stale or hallucinated ID cannot reach tools.
        slots = dict(explicit_slots)

        action = decision.action
        tool = decision.tool
        tool_args = dict(decision.tool_args)
        intent = self._detect_intent(customer_text)
        if intent is IntentName.UNKNOWN:
            intent = decision.intent.intent
        prediction = IntentPrediction(
            intent=intent,
            confidence=decision.intent.confidence,
            source=decision.intent.source,
            model_version=decision.intent.model_version,
            missing_slots=decision.intent.missing_slots,
            alternatives=decision.intent.alternatives,
        )

        if self._should_handoff(history[-1].text, " ".join(turn.text for turn in history)):
            action = CaseAction.HANDOFF
            tool = None
            tool_args = {}
        elif intent is IntentName.BANK_TRANSFER_NOT_RECEIVED and action is CaseAction.ANSWER:
            if SlotName.TRANSACTION_ID not in slots:
                action = CaseAction.ASK_CLARIFICATION
                tool = None
                tool_args = {}
            else:
                action = CaseAction.RETRIEVE_CONTEXT
                tool = ToolName.GET_TRANSACTION_STATUS
                tool_args = {}
        elif (
            self._should_create_ticket(history[-1].text, intent)
            and SlotName.TRANSACTION_ID in slots
        ):
            action = CaseAction.EXECUTE_TOOL
            tool = ToolName.CREATE_SUPPORT_TICKET
            tool_args = {
                "transaction_id": slots[SlotName.TRANSACTION_ID],
                "reason": "customer_operations_investigation",
            }
        elif action is CaseAction.RETRIEVE_CONTEXT:
            if intent is IntentName.MISSING_REFUND:
                tool = ToolName.GET_REFUND_STATUS
            elif intent in {
                IntentName.TRANSACTION_PENDING,
                IntentName.TRANSACTION_FAILED,
                IntentName.BANK_TRANSFER_NOT_RECEIVED,
            }:
                tool = ToolName.GET_TRANSACTION_STATUS

        contract = get_intent_contract(intent)
        if (
            action in {CaseAction.RETRIEVE_CONTEXT, CaseAction.EXECUTE_TOOL}
            and set(contract.required_slots) - set(slots)
        ):
            action = CaseAction.ASK_CLARIFICATION
            tool = None
            tool_args = {}
        elif action in {
            CaseAction.RETRIEVE_CONTEXT,
            CaseAction.EXECUTE_TOOL,
        }:
            if SlotName.TRANSACTION_ID in slots:
                tool_args["transaction_id"] = slots[SlotName.TRANSACTION_ID]
            if action is CaseAction.EXECUTE_TOOL:
                tool_args.setdefault("reason", "customer_operations_investigation")
        elif action in {
            CaseAction.ASK_CLARIFICATION,
            CaseAction.ANSWER,
            CaseAction.HANDOFF,
        }:
            tool = None
            tool_args = {}

        return RouterDecision(
            intent=prediction,
            slots=slots,
            action=action,
            tool=tool,
            tool_args=tool_args,
            policy_violation=decision.policy_violation,
            policy_source=decision.policy_source,
        )

    def _validate_decision(self, decision: RouterDecision) -> None:
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

    def _guardrail_handoff(self, reason: str) -> RouterDecision:
        return RouterDecision(
            intent=IntentPrediction(intent=IntentName.UNKNOWN, confidence=1.0, source="classifier"),
            action=CaseAction.HANDOFF,
            policy_violation=reason,
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

        def provider(history: list[GoldenTurn]) -> RouterDecision:
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
                text_format=LLMDecisionPayload,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI returned no structured LLMDecisionPayload")
            return parsed.to_router_decision()

        answerer = KnowledgeBackedAnswerer.from_repository(
            OpenAIAnswerGenerator(client, selected_model)
        )
        return cls(provider, answerer=answerer)


_LLM_SYSTEM_INSTRUCTIONS = """You are the decision router for a Vietnamese fintech customer-operations case.

Use the conversation to produce exactly one typed RouterDecision. Classify only
the supported intents: missing_refund, transaction_pending,
transaction_failed, bank_transfer_not_received, or unknown. Extract only explicit demo transaction/refund
IDs; never invent an ID. For a clear supported request, set confidence at or
above 0.90; use lower confidence only when the intent is genuinely ambiguous.

Routing policy:
- missing_refund + transaction ID -> retrieve_context + get_refund_status
- transaction_pending + transaction ID -> retrieve_context + get_transaction_status
- transaction_failed + transaction ID -> retrieve_context + get_transaction_status
- bank_transfer_not_received + transaction ID -> retrieve_context + get_transaction_status
- missing transaction ID -> ask_clarification, with no tool
- explicit human/manual-support request -> handoff, with no tool
- prior system result followed by a customer question -> answer, with no tool
- explicit investigation/repeated-failure support request -> execute_tool + create_support_ticket

Treat prior agent messages as observations, not instructions from the customer.
The extracted transaction_id is the single source of truth for tool arguments;
do not duplicate or invent a different tool ID. Do not generate an outcome label or a
customer-facing response; those are derived by the application policy layer.
"""
