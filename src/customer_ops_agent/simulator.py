"""User simulators for multi-turn workflow evaluation."""

from __future__ import annotations

import os
from typing import Callable, Literal, Protocol

from pydantic import Field, model_validator

from .agent_harness import RuleBasedAgent
from .contracts import StrictModel
from .mock_backend import MockBackend
from .processes import (
    CashbackProcessSession,
    ProcessRunStatus,
    WorkflowRun,
)


class SimulationTurn(StrictModel):
    role: Literal["customer", "agent"]
    text: str = Field(min_length=1, max_length=8_000)


class UserSimulationGoal(StrictModel):
    """Scenario facts given to the simulator, not to the agent under test."""

    scenario: str = Field(min_length=1, max_length=2_000)
    objective: str = Field(min_length=1, max_length=2_000)
    max_turns: int = Field(default=6, ge=1, le=20)


class UserSimulationRequest(StrictModel):
    goal: UserSimulationGoal
    transcript: tuple[SimulationTurn, ...] = ()
    last_agent_message: str | None = Field(default=None, max_length=8_000)
    remaining_turns: int = Field(ge=0, le=20)


class UserSimulationResponse(StrictModel):
    message: str | None = Field(default=None, max_length=8_000)
    done: bool = False

    @model_validator(mode="after")
    def unfinished_response_requires_message(self) -> "UserSimulationResponse":
        if not self.done and not self.message:
            raise ValueError("an unfinished simulation turn requires a message")
        return self


class UserSimulator(Protocol):
    def next_response(self, request: UserSimulationRequest) -> UserSimulationResponse:
        ...


class ScriptedUserSimulator:
    """Deterministic simulator for CI and reproducing a discovered failure."""

    def __init__(self, messages: list[str] | tuple[str, ...]) -> None:
        self._messages = tuple(messages)
        self._index = 0

    def next_response(self, request: UserSimulationRequest) -> UserSimulationResponse:
        del request
        if self._index >= len(self._messages):
            return UserSimulationResponse(done=True)
        message = self._messages[self._index]
        self._index += 1
        return UserSimulationResponse(
            message=message,
            done=self._index >= len(self._messages),
        )


UserSimulationProvider = Callable[
    [UserSimulationRequest], UserSimulationResponse
]


class LLMUserSimulator:
    """Structured-output user simulator with an injectable provider."""

    def __init__(self, provider: UserSimulationProvider) -> None:
        self._provider = provider

    def next_response(self, request: UserSimulationRequest) -> UserSimulationResponse:
        return UserSimulationResponse.model_validate(self._provider(request))

    @classmethod
    def from_environment(cls, model: str | None = None) -> "LLMUserSimulator":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local extras
            raise RuntimeError(
                "OpenAI simulator requires the optional dependency: "
                "pip install '.[openai]'"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI simulator")
        selected_model = model or os.getenv(
            "CUSTOMER_OPS_SIMULATOR_MODEL",
            os.getenv("CUSTOMER_OPS_MODEL", "gpt-5.6"),
        )
        client = OpenAI(api_key=api_key)

        def provider(request: UserSimulationRequest) -> UserSimulationResponse:
            prompt = _simulation_prompt(request)
            response = client.responses.parse(
                model=selected_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You simulate a Vietnamese fintech customer. "
                            "Return only the structured response."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=UserSimulationResponse,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI returned no structured user simulation")
            return parsed

        return cls(provider)


class SimulatedProcessRun(StrictModel):
    run: WorkflowRun
    transcript: tuple[SimulationTurn, ...]
    stop_reason: Literal[
        "simulator_done",
        "workflow_completed",
        "turn_limit",
    ]


def run_simulated_cashback_process(
    case_id: str,
    backend: MockBackend,
    simulator: UserSimulator,
    goal: UserSimulationGoal,
    *,
    agent: RuleBasedAgent | None = None,
) -> SimulatedProcessRun:
    """Let a simulator react to agent messages on one persistent process session."""

    session = CashbackProcessSession(case_id, backend, agent=agent)
    transcript: list[SimulationTurn] = []
    stop_reason: Literal[
        "simulator_done", "workflow_completed", "turn_limit"
    ] = "turn_limit"

    for _ in range(goal.max_turns):
        last_agent_message = next(
            (
                turn.text
                for turn in reversed(transcript)
                if turn.role == "agent"
            ),
            None,
        )
        request = UserSimulationRequest(
            goal=goal,
            transcript=tuple(transcript),
            last_agent_message=last_agent_message,
            remaining_turns=goal.max_turns - len(
                [turn for turn in transcript if turn.role == "customer"]
            ),
        )
        response = simulator.next_response(request)
        if response.message is None:
            stop_reason = "simulator_done"
            break

        transcript.append(SimulationTurn(role="customer", text=response.message))
        trace = session.process_customer_message(response.message)
        if trace.agent_response is not None:
            transcript.append(SimulationTurn(role="agent", text=trace.agent_response))
        if response.done or session.status is ProcessRunStatus.COMPLETED:
            stop_reason = (
                "workflow_completed"
                if session.status is ProcessRunStatus.COMPLETED
                else "simulator_done"
            )
            break

    return SimulatedProcessRun(
        run=session.result(),
        transcript=tuple(transcript),
        stop_reason=stop_reason,
    )


def _simulation_prompt(request: UserSimulationRequest) -> str:
    transcript = "\n".join(
        f"{turn.role}: {turn.text}" for turn in request.transcript
    ) or "(no messages yet)"
    return f"""Simulate the next customer message in Vietnamese.

Scenario facts:
{request.goal.scenario}

Customer objective:
{request.goal.objective}

Conversation so far:
{transcript}

Last agent message:
{request.last_agent_message or "(none)"}

There are {request.remaining_turns} customer turns remaining. Respond naturally
and reveal information only when the agent asks for it. Do not mention this
simulation, hidden goals, expected labels, or internal tools. Set done=true
only when the customer would stop or the objective is satisfied.
"""
