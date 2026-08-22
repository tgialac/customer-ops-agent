# Guardrails

The runtime uses deterministic application-level guardrails for the first
workflow. A framework such as NeMo Guardrails is not required at this stage;
the important boundaries are explicit, testable, and close to the tool and
policy code.

## Input

`check_input` rejects blank or overlong messages and a small set of explicit
instruction-hijacking patterns. A rejected message is handed off before any
backend tool is called. The failure is recorded in both the trace and
`CaseState.input_guardrail_failures`.

## Output

`check_output` is strict only for the source-backed bank-transfer workflow:

- an answer must use the active MoMo policy source;
- the response must be one of the bounded policy templates;
- clarification and handoff responses have fixed handles;
- every other customer-facing action fails closed to handoff.

Output failures are recorded in the trace and
`CaseState.output_guardrail_failures`. The guardrail is intentionally
application-owned so a future LLM answer layer can improve tone without
changing the allowed facts, time windows, or escalation boundary.

Component and runtime checks live in `tests/test_guardrails.py`.
