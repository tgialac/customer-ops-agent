"""Run every offline regression and source-backed workflow gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from customer_ops_agent.agent_harness import LLMAgent
from customer_ops_agent.eval_runner import run_evaluation


ROOT = Path(__file__).parents[1]
SUITES = (
    ("synthetic_baseline", "data/golden/customer_ops_golden_v1.jsonl"),
    ("bank_transfer_not_received", "data/golden/bank_transfer_not_received_v1.jsonl"),
    ("cashback_not_received", "data/golden/cashback_not_received_v1.jsonl"),
    ("google_play_refund", "data/golden/google_play_refund_v1.jsonl"),
    ("workflow_boundaries", "data/golden/workflow_boundaries_v1.jsonl"),
)


def run_release_gate(
    *,
    fixtures_path: Path = ROOT / "data/golden/fixtures.json",
    harness: str = "rule",
    model: str | None = None,
) -> dict[str, Any]:
    agent = LLMAgent.from_environment(model) if harness == "openai" else None
    reports: list[dict[str, Any]] = []
    for name, golden_path in SUITES:
        summary = run_evaluation(ROOT / golden_path, fixtures_path, agent)
        reports.append(
            {
                "name": name,
                "golden_set": golden_path,
                "total": summary.total,
                "passed": summary.passed,
                "pass_rate": summary.pass_rate,
                "failed_cases": [
                    record.case_id for record in summary.records if not record.passed
                ],
            }
        )

    total = sum(report["total"] for report in reports)
    passed = sum(report["passed"] for report in reports)
    return {
        "schema_version": 1,
        "harness": agent.harness_name if agent is not None else "rule_based_v1",
        "model": model if agent is not None else None,
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "passed_all": all(report["passed"] == report["total"] for report in reports),
        "suites": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=ROOT / "data/golden/fixtures.json"
    )
    parser.add_argument("--harness", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_release_gate(
        fixtures_path=args.fixtures,
        harness=args.harness,
        model=args.model,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed_all"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
