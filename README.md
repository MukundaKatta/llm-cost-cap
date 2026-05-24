# llm-cost-cap

[![PyPI](https://img.shields.io/pypi/v/llm-cost-cap.svg)](https://pypi.org/project/llm-cost-cap/)
[![Python](https://img.shields.io/pypi/pyversions/llm-cost-cap.svg)](https://pypi.org/project/llm-cost-cap/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Pre-flight USD cost gate for a single LLM call.**

Before sending a request, estimate the worst case cost (input tokens at
the input rate plus the requested `max_output_tokens` at the output rate)
and reject if it would exceed a configured per-call cap. This catches
the failure mode where an agent runs away and burns $20 on a single
call before the call ever leaves the process.

Zero runtime dependencies. Built-in price table covers Anthropic
Claude, OpenAI GPT-5, Google Gemini 2.5, and AWS Bedrock variants.

## Install

```bash
pip install llm-cost-cap
```

## Basic example

```python
from llm_cost_cap import CostCap, CostExceededError

cap = CostCap(max_usd_per_call=0.50)

try:
    cap.check(
        model="claude-opus-4-7",
        input_tokens=10_000,
        max_output_tokens=4_000,
    )
except CostExceededError as e:
    # don't send the call
    print(f"rejected {e.model}: ${e.estimated_usd:.4f} > ${e.cap_usd:.4f}")
```

`check()` returns the `EstimatedCost` breakdown on success so you can
log or budget against the exact number that gated the call:

```python
est = cap.check(
    model="claude-sonnet-4-6",
    input_tokens=2_000,
    max_output_tokens=500,
    cached_input_tokens=1_500,
)
est.total_usd          # 0.01395
est.input_usd          # 0.006
est.output_usd         # 0.0075
est.cached_input_usd   # 0.00045
```

## Wrap a call

`run()` does the check first and only invokes your function if the
estimate passes. Exceptions from your function propagate unchanged.

```python
import anthropic

client = anthropic.Anthropic()
cap = CostCap(max_usd_per_call=0.50)

result = cap.run(
    model="claude-opus-4-7",
    input_tokens=10_000,
    max_output_tokens=4_000,
    fn=lambda: client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4_000,
        messages=[{"role": "user", "content": prompt}],
    ),
)
```

If the cap would be exceeded the function is **not** called. No partial
work, no spend.

## Custom prices

Override the built-in table for private deployments, fine-tuned model
pricing, or just to keep the table fresh.

```python
from llm_cost_cap import CostCap, ModelPrice

prices = {
    "my-private-model": ModelPrice(
        input_per_million_usd=2.0,
        output_per_million_usd=8.0,
    ),
    "claude-opus-4-7": ModelPrice(
        input_per_million_usd=15.0,
        output_per_million_usd=75.0,
        cached_input_per_million_usd=1.50,
    ),
}
cap = CostCap(max_usd_per_call=1.0, prices=prices)
```

Or register one model into an existing cap:

```python
cap.add_model("homemade", ModelPrice(0.5, 1.5))
```

## Estimate without raising

`estimate()` returns the same `EstimatedCost` breakdown but never
raises on overage. Use it to log what a call would have cost, or to
decide between two model options.

```python
opus  = cap.estimate(model="opus",  input_tokens=5_000, max_output_tokens=2_000)
haiku = cap.estimate(model="haiku", input_tokens=5_000, max_output_tokens=2_000)
chosen = "opus" if opus.total_usd < cap.cap_usd else "haiku"
```

## Built-in price table

Prices are USD per million tokens as of 2026-05-24. Anthropic models
include published cached read rates.

| Family           | Models                                                  |
| ---------------- | ------------------------------------------------------- |
| Anthropic Claude | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5    |
| OpenAI GPT       | gpt-5.4, gpt-5, gpt-5-mini, gpt-5-nano                  |
| Google Gemini    | gemini-2.5-pro, gemini-2.5-flash                        |
| AWS Bedrock      | anthropic.* + meta.llama3-1-70b-instruct-v1:0           |

Short aliases (`opus`, `sonnet`, `haiku`, `gpt5`, `gemini-pro`,
`gemini-flash`) resolve to the canonical model id.

`builtin_prices()` returns a fresh copy of the table. `known_models()`
returns the sorted list of model ids (including aliases).

## What it does NOT do

- No HTTP. Doesn't talk to any LLM provider, doesn't count tokens for
  you. Bring the tokenizer of your choice (`tiktoken`, `anthropic.Client.count_tokens`,
  etc.) and pass the result in.
- No multi-call accounting. This caps a single call. For a running
  total of tokens / USD across many calls, use `token-budget-py`.
- No time windows. For a per-minute or per-day cap, use
  `llm-budget-window`.
- No live pricing fetch. The table is static; bump the package or pass
  your own `prices=` dict when a vendor changes pricing.

## Siblings

This is the per-call gate in a small family of LLM cost / budget
libraries:

- [`token-budget-py`](https://github.com/MukundaKatta/token-budget-py)
  for a running total of tokens / USD across many calls (fan-out
  workloads).
- [`llm-budget-window`](https://github.com/MukundaKatta/llm-budget-window)
  for a time-windowed cap (per minute / hour / day).
- [`claude-cost`](https://crates.io/crates/claude-cost) (Rust crate)
  for a more detailed cache-aware cost calculator.

## License

MIT
