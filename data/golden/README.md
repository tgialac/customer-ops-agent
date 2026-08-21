# MoMo golden set v1

This is a **synthetic, human-authored evaluation set** for the initial MoMo
Ops Agent slice. It is not sourced from MoMo customer data and does not encode
real MoMo policies, SLAs, or operational procedures.

## Coverage

- 60 cases total
- 20 `missing_refund`
- 20 `transaction_pending`
- 20 `transaction_failed`
- Vietnamese multi-turn and single-turn messages
- Missing-slot clarification
- Intent paraphrases and colloquial wording
- Tool selection expectations
- Stateful tool-result follow-ups
- Explicit human handoff cases

Each case includes the expected intent, slots, action, tool, resulting case
status, outcome, and challenge tags. The `expected_outcome` values describe
the test oracle for this project; they are not claims about production MoMo
behavior.

Validate it with:

```bash
python -m pytest -q
```
