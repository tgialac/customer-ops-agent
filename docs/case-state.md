# Case state and intent contract

The first slice supports three transaction-support intents:

| Intent | Allowed context | Required slot | Risk |
| --- | --- | --- | --- |
| `missing_refund` | transaction, refund | `transaction_id` | medium |
| `transaction_pending` | transaction | `transaction_id` | medium |
| `transaction_failed` | transaction | `transaction_id` | medium |

`unknown` is deliberately restricted to clarification or human handoff. It
cannot retrieve customer context or execute a tool.

## Design rules

- `CaseState` is the durable source of truth between triage, tools, and response.
- Unknown fields are rejected to prevent silent contract drift from LLM output.
- Context is intent-gated. A routed intent cannot access a scope outside its
  contract, and a context object cannot carry data without declaring its scope.
- `CaseState` updates are validated and return a new state instead of mutating
  the existing object.
- Intent confidence must meet the intent's threshold before it can be accepted.
- Case transitions are explicit; handoff requires a reason.
- `customer_ref` is a non-PII reference. Raw names, phone numbers, tokens, and
  account credentials do not belong in this contract.

The models expose Pydantic JSON Schema through `CaseState.model_json_schema()`
and `IntentContract.model_json_schema()` for structured model output and tool
boundaries.

## References

- [Monzo Ops Agent](https://monzo.com/blog/engineering-the-future-of-customer-operations-the-monzo-ops-agent): intent-gated context, triage/action logic, human handoff, and stateful workflow evaluation.
- [Anthropic — Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): start with simple composable workflows and add agentic complexity only when needed.
- [Anthropic — Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): treat context as a finite resource and curate it per step.
- [OpenAI — A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-agents/): structured tools, bounded runs, guardrails, and human intervention.
