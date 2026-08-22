# Grounded answer layer

The answer layer is deliberately downstream of routing, tools, and policy:

```text
customer message
  -> input guardrail
  -> router
  -> transaction-status tool
  -> deterministic policy
  -> active MoMo source retrieval
  -> structured LLM wording draft
  -> output guardrail
  -> answer or handoff
```

The model receives the customer message, the active source document, the
policy message key, and a deterministic fallback answer. It can improve
empathy and tone, but it cannot choose the message key, source document,
deadline, refund destination, or escalation path. Structured output enforces
the source/message identity, while the output guardrail checks mandatory facts
and rejects unsupported guarantees.

Runtime decisions keep an internal response handle for evaluation (for
example, `handoff` or `ask_for_transaction_id`) and expose a separate
customer-facing message. This prevents test handles from leaking into the
conversation while keeping routing outcomes deterministic.

An invalid draft is regenerated once with the previous draft included as a
failure signal. A second failure, unavailable source, or model/API error
fails closed to handoff. The deterministic policy response remains the safe
fallback when no answer model is configured.

This follows the production pattern described by Monzo: SME-vetted knowledge
for answer generation, self-correcting output guardrails, and separate
component/answer-generation/end-to-end evaluation. The implementation keeps
the current local lexical retriever; a vector store can replace it behind the
same `KnowledgeBackedAnswerer` boundary later.

## Human-review QA loop

The flagship review scope is `cashback_not_received_v1`, which is the closest
equivalent to Monzo's first missing-refund workflow. It is deliberately
reviewed as a separate human gate rather than being treated as complete just
because automated evals pass.

Run the source-backed suite through the deterministic harness for a fast local
check:

```bash
uv run python scripts/run_answer_qa.py --harness rule
```

Run it through the configured LLM to capture realistic wording for review:

```bash
uv run --env-file .env --extra openai python scripts/run_answer_qa.py \
  --harness openai --model gpt-5.6-luna
```

The command writes an ignored JSON artifact under `artifacts/qa/`. Each record
contains the customer-facing response, its internal response handle, source
URL/document, policy message key, retry count, automated checks, and
`review_status: "pending"`. A reviewer
can set that status to `approved` or `rejected` and add `reviewer_notes`.
Re-running with the same output path preserves review fields only when the
reviewed message and policy identity are unchanged; `--previous` can be used
to load a different prior artifact.

Run the explicit human-review gate after reviewing the artifact:

```bash
uv run python scripts/run_review_gate.py
```

See [the full review protocol](review-loop.md) and run the compact pitch demo
with `uv run python scripts/run_flagship_demo.py`.
