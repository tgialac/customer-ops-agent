# Evaluation runner and stateful mock backend

This iteration adds an offline baseline and an optional structured-output LLM
router while keeping the evaluator and backend unchanged.

## Components

- `mock_backend.py`: deterministic transaction/refund records, state version,
  audit log, and idempotent support-ticket creation.
- `agent_harness.py`: transparent rule-based baseline plus an injected-provider
  `LLMAgent`. Neither reads the expected labels from the golden set.
- `eval_runner.py`: replays every customer/agent turn and grades intent, slots,
  action, tool, case status, and outcome.
- `data/golden/fixtures.json`: backend state fixtures kept separate from the
  expected labels.

Run it with:

```bash
python -m momo_ops_agent.eval_runner
```

The current rule-based baseline passes the 60 synthetic cases. This is a
harness smoke test, not an LLM quality claim. The LLM adapter is intentionally
not called in CI unless a key is supplied; its provider can be injected in
unit tests.

The source-backed workflows have separate suites:

```bash
python -m momo_ops_agent.eval_runner \
  --golden-set data/golden/bank_transfer_not_received_v1.jsonl \
  --fixtures data/golden/fixtures.json
```

These cases grade the final action/status/outcome and the read-only lookup
recorded in the trace. This keeps policy acceptance separate from the broader
synthetic intent-routing baseline.

The cashback workflow is evaluated independently from the bank-transfer
workflow because its official policy has different eligibility, limits, and a
24-hour window:

```bash
uv run python -m momo_ops_agent.eval_runner \
  --golden-set data/golden/cashback_not_received_v1.jsonl \
  --fixtures data/golden/fixtures.json
```

For source-backed answer review, write a JSON artifact containing the final
customer response, source, policy key, guardrail checks, retry count, and
human-review status:

```bash
uv run --env-file .env --extra openai python scripts/run_answer_qa.py \
  --harness openai --model gpt-5.6-luna \
  --golden-set data/golden/cashback_not_received_v1.jsonl \
  --output artifacts/qa/cashback_not_received_v1_live.json
```

When the OpenAI harness is used, the same workflow also exercises the
knowledge-backed answer writer, structured draft validation, and one retry
before handoff. `AgentTrace.answer_generation_attempts` makes that behavior
observable.

Guardrail component and runtime tests cover both rejection paths: input
instruction hijacking must not call a backend tool, and an unapproved
source-backed answer must become a handoff. See [the guardrail design](guardrails.md).

LLM iteration uses `scripts/run_smoke_eval.py`, a fixed 12-case stratified
gate covering missing-slot, retrieval, answer, ticket, handoff, and ambiguous
routing behavior. The 60-case set is reserved for regression/release checks.

Run the LLM adapter with:

```bash
python -m pip install -e '.[test,openai]'
export OPENAI_API_KEY='...'
python -m momo_ops_agent.eval_runner --harness openai --model gpt-5.6
```

The adapter uses the OpenAI Responses API's Pydantic Structured Outputs to
parse only `RouterDecision` (intent, slots, action, and tool arguments). The
application then enforces the intent contract, dispatches only the three
allowlisted backend tools, and derives the outcome/response outside the model.
A semantic contract violation becomes a visible safe handoff in the trace, so
it fails the golden case without mutating backend state.

## Why this shape

Monzo describes layered component, answer-generation, and end-to-end
evaluation, plus a stateful simulated environment for tool calls. Anthropic's
agent-evaluation guidance similarly separates the agent harness from the
evaluation harness and recommends grading both the trace and final environment
outcome.

- [Monzo Ops Agent](https://monzo.com/blog/engineering-the-future-of-customer-operations-the-monzo-ops-agent)
- [Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [OpenAI — Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI — Create a model response](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
