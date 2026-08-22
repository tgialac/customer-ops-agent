# Human review loop for the flagship workflow

The flagship demo is `cashback_not_received_v1`. It is intentionally narrow
and source-backed, with four important branches: missing transaction ID,
cashback within 24 hours, overdue handoff, and cashback-account limit.

## Run the review pack

Generate customer-facing answers with the deterministic harness:

```bash
uv run python scripts/run_answer_qa.py --harness rule
```

For an LLM wording pass, use the configured model:

```bash
uv run --env-file .env --extra openai python scripts/run_answer_qa.py \
  --harness openai --model gpt-5.6-luna
```

The output is `artifacts/qa/cashback_not_received_v1.json`. Re-running the
command preserves existing `review_status` and `reviewer_notes` by case ID.

## Reviewer checklist

For every record, a reviewer checks:

1. The customer-facing response addresses the actual customer message.
2. The policy facts match the linked official source.
3. The response does not invent a deadline, refund, or transaction action.
4. The action is appropriate: answer, ask for the transaction ID, or handoff.
5. The response is safe and understandable in Vietnamese.

The reviewer then changes `review_status` from `pending` to `approved` or
`rejected` and records the reason in `reviewer_notes`.

## Release gate

The artifact is not review-complete until every case passes automated checks
and has explicit human approval:

```bash
uv run python scripts/run_review_gate.py
```

Pending or rejected cases make the command exit non-zero. This keeps an
automated `93/93` regression pass separate from the human quality decision;
the current repository does not claim SME approval before someone performs
the review.

## Pitch demo

Show the four representative branches with:

```bash
uv run python scripts/run_flagship_demo.py
```

This prints the customer message, chosen action, outcome, tool result, and
guardrail status without requiring a real customer account or API.
