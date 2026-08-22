# MoMo Ops Agent

An evaluation-first customer operations agent inspired by the [Monzo Ops Agent](https://monzo.com/blog/engineering-the-future-of-customer-operations-the-monzo-ops-agent), designed for fintech customer support workflows in Vietnamese.

## Architecture

![MoMo Ops Agent architecture](docs/architecture.png)

## Status

The first typed contract is implemented in `src/momo_ops_agent/contracts.py`:

- Versioned `CaseState` with explicit transitions and timezone-aware timestamps
- Intent catalog with confidence thresholds, context scopes, risk levels, and allowed actions
- Intent-gated transaction/refund context with no raw PII fields
- Pydantic validation and JSON Schema support

See [the contract design](docs/case-state.md) for the initial scope and design references.

## External intent benchmark

Prepare the filtered [BANKING77](https://huggingface.co/datasets/PolyAI/banking77)
subset with:

```bash
python -m pip install -e .
python scripts/prepare_banking77.py
```

The output is an English single-turn benchmark mapped to the initial project
intents. It is intentionally kept separate from the Vietnamese synthetic MoMo
golden set at [`data/golden/momo_golden_v1.jsonl`](data/golden/momo_golden_v1.jsonl),
which evaluates state transitions, tool calls, and outcomes.

## Evaluation baseline

Run the offline rule-based harness and stateful backend with:

```bash
python -m momo_ops_agent.eval_runner
```

See [the evaluation design](docs/evaluation.md) for the trace, grader, and
mock-backend boundaries.

The same runner can execute the optional structured-output LLM decision
adapter. Install the extra and provide a key outside the repository:

```bash
python -m pip install -e '.[test,openai]'
export OPENAI_API_KEY='...'
python -m momo_ops_agent.eval_runner --harness openai --model gpt-5.6
```

The model proposes a `RouterDecision` only. The application validates the
intent contract and allowlisted tool arguments, runs the stateful backend, and
materializes the outcome/response through deterministic policy code.

For fast LLM iteration, run the 12-case smoke gate instead of the full
regression set:

```bash
uv run --extra openai python scripts/run_smoke_eval.py --model gpt-5.6-luna
```
