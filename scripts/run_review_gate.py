"""Require automated QA and explicit human approval for a review artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_ops_agent.qa import AnswerQAReport, evaluate_review_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/qa/cashback_not_received_v1.json"),
    )
    args = parser.parse_args()
    if not args.report.exists():
        raise SystemExit(f"review artifact not found: {args.report}")

    report = AnswerQAReport.model_validate(json.loads(args.report.read_text(encoding="utf-8")))
    gate = evaluate_review_gate(report)
    print(json.dumps(gate.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if not gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
