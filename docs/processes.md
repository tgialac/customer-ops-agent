# Executable processes

The first executable process is
`data/processes/cashback_not_received_v2.md`. Its Markdown front matter defines
the workflow name, version, entry intent, and ordered steps. The body remains
human-readable documentation; only the validated metadata is loaded by the
runtime.

`run_cashback_process` preserves one `CaseState`, one simulated backend, and a
conversation history across customer turns. Each turn records:

- the action and outcome selected by the existing agent runtime;
- the tool call, if any;
- the plan before and after the turn;
- the evidence attached to completed steps.

The process layer does not change policy decisions or allow new financial
actions. It orchestrates the existing bounded tools and keeps ticket creation
idempotent. A successful cashback answer within 24 hours waits for an external
result; a policy answer resolves the case; an overdue or failed lookup hands
off. An explicit customer request can resume a handed-off case to create a
support ticket.

Run the process suite as part of the release gate:

```bash
uv run python scripts/run_release_gate.py
```

Show the evolving plan for a representative conversation:

```bash
uv run python scripts/run_process_demo.py
```

## User simulation

`run_simulated_cashback_process` feeds each generated customer message into a
single process session, then gives the resulting customer-facing agent answer
back to the simulator on the next turn. `ScriptedUserSimulator` is used for
deterministic CI and failure reproduction. `LLMUserSimulator` uses structured
output and an explicit goal, but the goal is kept outside the agent's context;
the model under test only sees the simulated conversation.

The LLM simulator is an optional experiment harness, not a production customer
model. Its outputs still go through the same input guardrails, policy engine,
tool allowlist, and workflow state grader.
