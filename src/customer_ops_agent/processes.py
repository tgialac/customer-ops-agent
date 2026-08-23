"""Executable, multi-turn process orchestration for bounded workflows."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field

from .agent_harness import AgentTrace, RuleBasedAgent
from .contracts import (
    CaseState,
    CaseStatus,
    IntentName,
    RefundStatus,
    SlotName,
    StrictModel,
    TransactionStatus,
)
from .evaluation import GoldenTurn, GoldenTurnRole, ToolName
from .mock_backend import MockBackend, TransactionRecord


class ProcessStepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ProcessRunStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    WAITING_FOR_CUSTOMER = "waiting_for_customer"
    WAITING_EXTERNAL = "waiting_external"
    COMPLETED = "completed"
    HANDED_OFF = "handed_off"


class ProcessDefinition(StrictModel):
    """Human-readable process metadata loaded from Markdown front matter."""

    schema_version: Literal[1] = 1
    workflow: str = Field(pattern=r"^[a-z0-9][a-z0-9_]+$")
    version: int = Field(ge=1)
    entry_intent: IntentName
    steps: tuple[str, ...] = Field(min_length=1)


class ProcessStep(StrictModel):
    step_id: str = Field(min_length=1)
    status: ProcessStepStatus = ProcessStepStatus.PENDING
    evidence: tuple[str, ...] = ()


class WorkflowPlan(StrictModel):
    workflow: str = Field(min_length=1)
    version: int = Field(ge=1)
    steps: tuple[ProcessStep, ...] = Field(min_length=1)
    current_step: str | None = None

    @classmethod
    def from_definition(cls, definition: ProcessDefinition) -> "WorkflowPlan":
        steps = tuple(ProcessStep(step_id=step_id) for step_id in definition.steps)
        return cls(
            workflow=definition.workflow,
            version=definition.version,
            steps=steps,
            current_step=steps[0].step_id,
        )


class ProcessTurnTrace(StrictModel):
    turn_index: int = Field(ge=0)
    customer_message: str = Field(min_length=1)
    agent_response: str | None = None
    action: str
    outcome: str
    tool_sequence: tuple[ToolName, ...] = ()
    plan_before: WorkflowPlan
    plan_after: WorkflowPlan


class WorkflowRun(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str = Field(min_length=1)
    workflow: str = Field(min_length=1)
    status: ProcessRunStatus
    case_state: CaseState
    plan: WorkflowPlan
    traces: tuple[ProcessTurnTrace, ...] = ()
    backend_snapshot: dict[str, object] = Field(default_factory=dict)

    @property
    def final_trace(self) -> ProcessTurnTrace:
        if not self.traces:
            raise ValueError("workflow run has no turns")
        return self.traces[-1]


class CashbackProcessSession:
    """Mutable orchestration session used by interactive/user-simulated runs."""

    def __init__(
        self,
        case_id: str,
        backend: MockBackend,
        *,
        agent: RuleBasedAgent | None = None,
        process_path: Path | None = None,
    ) -> None:
        definition = load_process(
            process_path or _default_cashback_process_path()
        )
        if definition.entry_intent is not IntentName.MISSING_REFUND:
            raise ValueError("cashback process must enter through missing_refund")
        self.case_id = case_id
        self.workflow = definition.workflow
        self.backend = backend
        self.agent = agent or RuleBasedAgent()
        self.state = CaseState(customer_ref=f"process_{case_id}")
        self.plan = WorkflowPlan.from_definition(definition)
        self.history: list[GoldenTurn] = []
        self.traces: list[ProcessTurnTrace] = []
        self.status = ProcessRunStatus.IN_PROGRESS

    def process_customer_message(self, message: str) -> ProcessTurnTrace:
        turn = GoldenTurn(role=GoldenTurnRole.CUSTOMER, text=message)
        self.history.append(turn)
        self.state, agent_trace = self.agent.run_turn(
            self.state,
            self.history,
            len(self.traces),
            self.backend,
        )
        plan_before = self.plan
        self.plan, self.status = _advance_plan(
            self.plan, self.status, agent_trace
        )
        process_trace = ProcessTurnTrace(
            turn_index=len(self.traces),
            customer_message=message,
            agent_response=agent_trace.decision.customer_response,
            action=agent_trace.decision.action.value,
            outcome=agent_trace.decision.outcome.value,
            tool_sequence=(
                (agent_trace.tool_result.tool_name,)
                if agent_trace.tool_result is not None
                else ()
            ),
            plan_before=plan_before,
            plan_after=self.plan,
        )
        self.traces.append(process_trace)
        if agent_trace.decision.customer_response:
            self.history.append(
                GoldenTurn(
                    role=GoldenTurnRole.AGENT,
                    text=agent_trace.decision.customer_response,
                )
            )
        return process_trace

    def result(self) -> WorkflowRun:
        if not self.traces:
            raise ValueError("cashback process requires at least one customer message")
        return WorkflowRun(
            case_id=self.case_id,
            workflow=self.workflow,
            status=self.status,
            case_state=self.state,
            plan=self.plan,
            traces=tuple(self.traces),
            backend_snapshot=self.backend.snapshot(),
        )


class WorkflowExpectation(StrictModel):
    final_process_status: ProcessRunStatus
    final_case_status: CaseStatus
    tool_sequence: tuple[ToolName, ...] = ()
    ticket_count: int = Field(default=0, ge=0)


class WorkflowGrade(StrictModel):
    case_id: str
    checks: dict[str, bool]
    passed: bool


class CashbackProcessCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    messages: tuple[str, ...] = Field(min_length=1)
    transaction_id: str | None = None
    transaction_present: bool = True
    elapsed_hours: int | None = Field(default=12, ge=0)
    cashback_reason: str | None = None
    expected: WorkflowExpectation


class ProcessEvalSummary(StrictModel):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    grades: tuple[WorkflowGrade, ...]


def load_process(path: Path) -> ProcessDefinition:
    """Load a small process definition without making Markdown executable code."""

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"process is missing front matter: {path}")
    _, raw_metadata, _ = text.split("---\n", 2)
    metadata: dict[str, str] = {}
    for line in raw_metadata.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid process front matter in {path}: {line}")
        metadata[key.strip()] = value.strip()

    raw_steps = tuple(
        step.strip()
        for step in metadata.get("steps", "").split(",")
        if step.strip()
    )
    return ProcessDefinition(
        workflow=metadata["workflow"],
        version=int(metadata["version"]),
        entry_intent=IntentName(metadata["entry_intent"]),
        steps=raw_steps,
    )


def load_cashback_process_cases(path: Path) -> list[CashbackProcessCase]:
    return [
        CashbackProcessCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_cashback_process_evaluation(
    path: Path,
    *,
    agent: RuleBasedAgent | None = None,
) -> ProcessEvalSummary:
    grades: list[WorkflowGrade] = []
    for case in load_cashback_process_cases(path):
        backend = MockBackend()
        if case.transaction_id is not None and case.transaction_present:
            backend = MockBackend(
                [
                    TransactionRecord(
                        transaction_id=case.transaction_id,
                        status=TransactionStatus.COMPLETED,
                        refund_id=f"refund_for_{case.transaction_id}",
                        refund_status=RefundStatus.PROCESSING,
                        cashback_elapsed_hours=case.elapsed_hours,
                        cashback_reason=case.cashback_reason,
                    )
                ]
            )
        run = run_cashback_process(
            case.case_id,
            case.messages,
            backend,
            agent=agent,
        )
        grades.append(grade_workflow_run(run, case.expected))
    passed = sum(grade.passed for grade in grades)
    return ProcessEvalSummary(
        total=len(grades),
        passed=passed,
        pass_rate=passed / len(grades) if grades else 0.0,
        grades=tuple(grades),
    )


def run_cashback_process(
    case_id: str,
    customer_messages: Iterable[str],
    backend: MockBackend,
    *,
    agent: RuleBasedAgent | None = None,
    process_path: Path | None = None,
) -> WorkflowRun:
    """Run the cashback process while preserving state across customer turns."""

    session = CashbackProcessSession(
        case_id,
        backend,
        agent=agent,
        process_path=process_path,
    )
    for message in customer_messages:
        session.process_customer_message(message)
    return session.result()


def grade_workflow_run(
    run: WorkflowRun, expectation: WorkflowExpectation
) -> WorkflowGrade:
    """Grade both conversation outcome and final simulated environment state."""

    actual_tools = tuple(
        tool
        for trace in run.traces
        for tool in trace.tool_sequence
    )
    ticket_count = len(run.backend_snapshot.get("audit_log", []))
    checks = {
        "process_status": run.status is expectation.final_process_status,
        "case_status": run.case_state.status is expectation.final_case_status,
        "tool_sequence": actual_tools == expectation.tool_sequence,
        "ticket_count": ticket_count == expectation.ticket_count,
        "plan_step_consistent": (
            not any(
                step.status is ProcessStepStatus.ACTIVE for step in run.plan.steps
            )
            if run.status in {ProcessRunStatus.COMPLETED, ProcessRunStatus.HANDED_OFF}
            else run.plan.current_step is None
            or any(
                step.step_id == run.plan.current_step
                and step.status is ProcessStepStatus.ACTIVE
                for step in run.plan.steps
            )
        ),
    }
    return WorkflowGrade(
        case_id=run.case_id,
        checks=checks,
        passed=all(checks.values()),
    )


def _advance_plan(
    plan: WorkflowPlan,
    current_status: ProcessRunStatus,
    trace: AgentTrace,
) -> tuple[WorkflowPlan, ProcessRunStatus]:
    statuses = {step.step_id: step for step in plan.steps}
    decision = trace.decision
    _complete(statuses, "identify_issue", decision.outcome.value)

    if decision.slots.get(SlotName.TRANSACTION_ID) is None:
        if decision.action.value == "handoff":
            _complete(statuses, "finish_or_handoff", decision.outcome.value)
            return _rebuild_plan(plan, statuses, None), ProcessRunStatus.HANDED_OFF
        _activate(statuses, "collect_transaction_id")
        return _rebuild_plan(plan, statuses, "collect_transaction_id"), ProcessRunStatus.WAITING_FOR_CUSTOMER

    _complete(statuses, "collect_transaction_id", "transaction_id_verified")
    if trace.tool_result is not None and trace.tool_result.tool_name is ToolName.GET_REFUND_STATUS:
        if trace.tool_result.success:
            _complete(statuses, "retrieve_refund_status", "refund_status_retrieved")
        else:
            _complete(statuses, "retrieve_refund_status", trace.tool_result.error_code or "lookup_failed")

    if decision.policy_message_key is not None:
        _complete(statuses, "apply_policy", decision.policy_message_key)

    if trace.tool_result is not None and trace.tool_result.tool_name is ToolName.CREATE_SUPPORT_TICKET:
        if trace.tool_result.success:
            _complete(statuses, "create_support_ticket", "ticket_created")
            _activate(statuses, "finish_or_handoff")
            return _rebuild_plan(plan, statuses, "finish_or_handoff"), ProcessRunStatus.WAITING_EXTERNAL

    if decision.action.value == "handoff":
        _complete(statuses, "finish_or_handoff", decision.outcome.value)
        return _rebuild_plan(plan, statuses, None), ProcessRunStatus.HANDED_OFF
    if decision.action.value == "answer":
        _complete(statuses, "finish_or_handoff", decision.outcome.value)
        if decision.policy_message_key == "cashback_pending_within_24_hours":
            return _rebuild_plan(plan, statuses, None), ProcessRunStatus.WAITING_EXTERNAL
        return _rebuild_plan(plan, statuses, None), ProcessRunStatus.COMPLETED

    _activate(statuses, "retrieve_refund_status")
    return _rebuild_plan(plan, statuses, "retrieve_refund_status"), current_status


def _complete(steps: dict[str, ProcessStep], step_id: str, evidence: str) -> None:
    step = steps.get(step_id)
    if step is None:
        return
    evidence_values = (*step.evidence, evidence)
    steps[step_id] = step.model_copy(
        update={"status": ProcessStepStatus.COMPLETED, "evidence": evidence_values}
    )


def _activate(steps: dict[str, ProcessStep], step_id: str) -> None:
    step = steps.get(step_id)
    if step is None or step.status is ProcessStepStatus.ACTIVE:
        return
    steps[step_id] = step.model_copy(update={"status": ProcessStepStatus.ACTIVE})


def _rebuild_plan(
    plan: WorkflowPlan,
    steps: dict[str, ProcessStep],
    current_step: str | None,
) -> WorkflowPlan:
    return plan.model_copy(
        update={
            "steps": tuple(steps[step.step_id] for step in plan.steps),
            "current_step": current_step,
        }
    )


def _default_cashback_process_path() -> Path:
    return Path(__file__).parents[2] / "data" / "processes" / "cashback_not_received_v2.md"


def workflow_grade_json(grade: WorkflowGrade) -> str:
    """Stable JSON helper for small demos and CI artifacts."""

    return json.dumps(grade.model_dump(mode="json"), ensure_ascii=False, indent=2)
