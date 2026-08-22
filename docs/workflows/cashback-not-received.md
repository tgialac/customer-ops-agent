# Workflow 2: cashback not received

This workflow covers promotional cashback on an eligible MoMo payment. It is
separate from merchant refunds, bank-transfer reversals, and Google Play
refunds because those flows have different policies and destinations.

## Source-backed behavior

The official MoMo FAQ says:

- some services are not eligible for cashback, including transfers, red
  envelopes, withdrawals, top-ups, point-of-service deposits/withdrawals, and
  Visa/Mastercard payments;
- the cashback account has a 12,000,000 VND balance limit;
- the monthly cashback limit is 2,000,000 VND;
- when the system is slower than expected, the customer should wait up to 24
  hours and then submit a Help request with the transaction code.

The runtime only uses these branches when the refund-status tool returns the
corresponding verified cashback facts. It does not infer eligibility from the
customer's wording. An overdue case hands off to Help rather than promising a
credit date.

Source: [MoMo FAQ: why cashback was not received](https://www.momo.vn/hoi-dap/tai-sao-toi-khong-duoc-hoan-tien-khi-thanh-toan-dich-vu-nay)

The source-backed regression suite is
`data/golden/cashback_not_received_v1.jsonl`.
