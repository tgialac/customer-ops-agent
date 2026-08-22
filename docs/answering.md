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

An invalid draft is regenerated once with the previous draft included as a
failure signal. A second failure, unavailable source, or model/API error
fails closed to handoff. The deterministic policy response remains the safe
fallback when no answer model is configured.

This follows the production pattern described by Monzo: SME-vetted knowledge
for answer generation, self-correcting output guardrails, and separate
component/answer-generation/end-to-end evaluation. The implementation keeps
the current local lexical retriever; a vector store can replace it behind the
same `KnowledgeBackedAnswerer` boundary later.
