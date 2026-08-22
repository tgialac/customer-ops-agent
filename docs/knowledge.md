# Source-backed MoMo knowledge base

The initial corpus under `data/knowledge/momo/` is based on public MoMo pages
checked on 2026-08-22. Documents are intentionally split by product scope:

- `bank_transfer_reversal`: pending bank transfers and returned funds.
- `cashback_not_received`: promotional cashback, not general payment refunds.
- `merchant_refund`: MoMo Payment API documentation for merchants only.
- `google_play_refund`: app purchases handled through Google Play.

The corpus does not invent a universal refund SLA. A document carries its
source URL, audience, scope, checked date, and status. Future or archived
documents are excluded from retrieval. In particular, the general MoMo terms
page captured on 2026-08-22 states a future effective date of 2026-08-27 and
is retained for tracking only.

`KnowledgeStore` currently uses deterministic lexical retrieval. Its interface
is intentionally small so a reviewed vector-store implementation can replace
it later without changing policy filtering or answer-layer callers.
