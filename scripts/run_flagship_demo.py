"""Show the flagship cashback workflow across its main customer outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from customer_ops_agent.agent_harness import RuleBasedAgent
from customer_ops_agent.eval_runner import build_backend, load_cases, load_fixture_config


ROOT = Path(__file__).parents[1]


def main() -> None:
    cases = load_cases(ROOT / "data/golden/cashback_not_received_v1.jsonl")
    fixtures = load_fixture_config(ROOT / "data/golden/fixtures.json")
    selected_ids = {
        "cashback-v1-001",  # missing transaction ID
        "cashback-v1-002",  # within the 24-hour window
        "cashback-v1-003",  # overdue handoff
        "cashback-v1-005",  # account limit
    }
    agent = RuleBasedAgent()
    demo = []
    for case in cases:
        if case.case_id not in selected_ids:
            continue
        run = agent.run(case.case_id, case.turns, build_backend(case, fixtures))
        trace = run.trace[-1]
        demo.append(
            {
                "case_id": case.case_id,
                "customer_message": case.turns[-1].text,
                "action": run.final_decision.action.value,
                "outcome": run.final_decision.outcome.value,
                "customer_response": run.final_decision.customer_response,
                "tool": trace.tool_result.tool_name.value if trace.tool_result else None,
                "tool_success": trace.tool_result.success if trace.tool_result else None,
                "guardrails": {
                    "input": trace.input_guardrail.passed if trace.input_guardrail else None,
                    "output": trace.output_guardrail.passed if trace.output_guardrail else None,
                },
            }
        )
    print(json.dumps(demo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
