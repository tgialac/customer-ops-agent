"""Run answer-generation QA and write a human-reviewable JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from momo_ops_agent.agent_harness import LLMAgent, RuleBasedAgent
from momo_ops_agent.qa import run_answer_qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=Path("data/golden/bank_transfer_not_received_v1.jsonl"),
    )
    parser.add_argument(
        "--fixtures", type=Path, default=Path("data/golden/fixtures.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qa/bank_transfer_not_received_v1.json"),
    )
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--harness", choices=("rule", "openai"), default="openai")
    parser.add_argument("--model", default="gpt-5.6-luna")
    args = parser.parse_args()

    agent = (
        LLMAgent.from_environment(args.model)
        if args.harness == "openai"
        else RuleBasedAgent()
    )
    report = run_answer_qa(
        args.golden_set,
        args.fixtures,
        agent,
        previous_path=args.previous,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "harness": report.harness,
                "total": report.total,
                "automated_passed": report.automated_passed,
                "review_pending": sum(
                    record.review_status == "pending" for record in report.records
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
