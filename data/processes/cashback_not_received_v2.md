---
workflow: cashback_not_received_v2
version: 1
entry_intent: missing_refund
steps: identify_issue, collect_transaction_id, retrieve_refund_status, apply_policy, create_support_ticket, finish_or_handoff
---

# Cashback not received — executable process

This process is intentionally bounded to verified cashback facts. The agent
must not infer eligibility, create a financial refund, or promise a credit
date from the customer's message alone.

## Steps

1. Identify the cashback issue and keep the workflow separate from merchant
   refunds, bank-transfer reversals, and Google Play refunds.
2. Collect and validate the transaction ID before reading customer context.
3. Retrieve the cashback status from the allowlisted backend tool.
4. Apply the deterministic cashback policy to the verified result.
5. Create a support ticket only when the customer explicitly asks for support
   or the process requires an investigation.
6. Finish with a bounded answer, wait for an external result, or hand off.
