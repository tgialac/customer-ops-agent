# Workflow 1: bank transfer not received

This is the first concrete customer-operations workflow for the project. It
is intentionally narrower than the existing synthetic `missing_refund` intent:
the customer says a transfer from MoMo to a bank account/card was debited, but
the beneficiary has not received the money.

## In scope

- transfer status is pending and awaiting reconciliation;
- transfer is successful but the beneficiary bank has not posted the money yet;
- transfer failed and the money is returning to MoMo or the funding bank;
- wrong bank details, which require support-led recovery;
- missing transaction ID, which requires clarification.

## Out of scope

- MoMo-to-MoMo transfer to an unregistered recipient;
- bank top-up or withdrawal pending;
- promotional cashback;
- merchant Payment API refund;
- Google Play app-purchase refund.

These flows have different public instructions and must not share one generic
refund answer.

## Source-backed behavior

- For a pending MoMo-to-bank transfer, the public FAQ describes a 1–2 working
  day reconciliation window and directs the customer to Help if there is still
  no result after that window.
- For a successful transfer, the beneficiary bank may take 1–3 working days to
  post the money; after that the customer should submit a Help request.
- For a failed transfer, the money is returned to MoMo or the bank depending on
  the funding source. A bank-side return may take another 1–2 working days.
- For wrong details, recovery depends on the receiving bank and potentially the
  mistaken recipient; this is a support/handoff path, not an automatic refund.

The deterministic decision table is implemented in
`src/momo_ops_agent/policies.py`. It treats the public windows as escalation
thresholds, not guaranteed settlement times: a pending transfer is only
explained within 2 working days, a successful transfer with delayed posting
within 3 working days, and a failed transfer return within 2 working days.
Missing timing data does not create a new promise; an unknown state or an
overdue case goes to support. The runtime calls the transaction-status tool
first, stores only the intent-approved transaction context, and renders one of
the bounded Vietnamese response templates from the policy decision. A failed
lookup also hands off without answering from incomplete facts.

The source-backed regression suite is
`data/golden/bank_transfer_not_received_v1.jsonl`; unlike the 60-case
synthetic baseline, its expected outcomes are tied to this workflow's cited
MoMo policy boundaries.

Sources:

- [MoMo: money debited but bank beneficiary has not received it](https://www.momo.vn/hoi-dap/vi-sao-tai-khoan-da-bi-tru-tien-nhung-tai-khoan-ngan-hang-nguoi-nhan-chua-nhan-duoc)
- [MoMo: pending top-up/withdrawal](https://www.momo.vn/hoi-dap/tai-khoan-bi-tru-tien-nhung-giao-dich-dang-cho-xu-ly)
- [MoMo: transfer to an unregistered MoMo recipient](https://www.momo.vn/hoi-dap/toi-co-the-chuyen-tien-cho-nguoi-chua-co-tai-khoan-momo-khong)

The second and third sources are boundary references: they explain why those
cases are excluded from this workflow.
