"""Run the small, representative LLM smoke gate.

This intentionally uses 12 cases instead of the full regression set. The
smoke gate is for fast contract/policy iteration; the full offline release
gate lives in ``scripts/run_release_gate.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_ops_agent.agent_harness import LLMAgent
from customer_ops_agent.eval_runner import evaluate_case, load_cases, load_fixture_config


ROOT = Path(__file__).parents[1]
SMOKE_CASE_IDS = frozenset(
    {
        "customer-ops-golden-v1-001",
        "customer-ops-golden-v1-002",
        "customer-ops-golden-v1-007",
        "customer-ops-golden-v1-009",
        "customer-ops-golden-v1-021",
        "customer-ops-golden-v1-022",
        "customer-ops-golden-v1-025",
        "customer-ops-golden-v1-031",
        "customer-ops-golden-v1-041",
        "customer-ops-golden-v1-042",
        "customer-ops-golden-v1-047",
        "customer-ops-golden-v1-060",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-luna")
    args = parser.parse_args()

    cases = [
        case
        for case in load_cases(ROOT / "data/golden/customer_ops_golden_v1.jsonl")
        if case.case_id in SMOKE_CASE_IDS
    ]
    fixtures = load_fixture_config(ROOT / "data/golden/fixtures.json")
    agent = LLMAgent.from_environment(args.model)
    records = [evaluate_case(case, fixtures, agent) for case in cases]
    passed = sum(record.passed for record in records)
    print(
        json.dumps(
            {
                "harness": agent.harness_name,
                "model": args.model,
                "total": len(records),
                "passed": passed,
                "pass_rate": passed / len(records) if records else 0.0,
                "failed_cases": [record.case_id for record in records if not record.passed],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
