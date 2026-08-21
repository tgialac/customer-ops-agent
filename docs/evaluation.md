# Evaluation runner and stateful mock backend

This iteration adds an offline baseline so the agent can be measured before an
LLM is introduced.

## Components

- `mock_backend.py`: deterministic transaction/refund records, state version,
  audit log, and idempotent support-ticket creation.
- `agent_harness.py`: transparent rule-based baseline. It does not read the
  expected labels from the golden set.
- `eval_runner.py`: replays every customer/agent turn and grades intent, slots,
  action, tool, case status, and outcome.
- `data/golden/fixtures.json`: backend state fixtures kept separate from the
  expected labels.

Run it with:

```bash
python -m momo_ops_agent.eval_runner
```

The current rule-based baseline passes the 60 synthetic cases. This is a
harness smoke test, not an LLM quality claim. The next agent implementation can
replace `RuleBasedAgent` while keeping the same runner and graders.

## Why this shape

Monzo describes layered component, answer-generation, and end-to-end
evaluation, plus a stateful simulated environment for tool calls. Anthropic's
agent-evaluation guidance similarly separates the agent harness from the
evaluation harness and recommends grading both the trace and final environment
outcome.

- [Monzo Ops Agent](https://monzo.com/blog/engineering-the-future-of-customer-operations-the-monzo-ops-agent)
- [Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
