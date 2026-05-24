"""llm-cost-cap - pre-flight USD cost gate for a single LLM call.

Before sending a request, estimate the worst-case cost (input tokens at
the input rate plus the requested ``max_output_tokens`` at the output
rate) and reject if it would exceed a configured per-call cap. Catches
the "agent went and burned $20 on one call" failure mode before the
call leaves the process.

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
        log.warning("rejected %s: $%.4f > $%.4f", e.model, e.estimated_usd, e.cap_usd)

Or wrap the call directly:

    result = cap.run(
        model="claude-opus-4-7",
        input_tokens=10_000,
        max_output_tokens=4_000,
        fn=lambda: client.messages.create(...),
    )

This is the per-call gate. Pair it with:

  * ``token-budget-py`` for a running total of tokens / USD across many calls
  * ``llm-budget-window`` for a time-windowed cap (per minute / hour / day)
  * ``claude-cost`` (Rust crate) for a cache-aware cost calculator
"""

from llm_cost_cap.cap import (
    CostCap,
    CostExceededError,
    EstimatedCost,
    UnknownModelError,
)
from llm_cost_cap.prices import ModelPrice, builtin_prices, known_models

__version__ = "0.1.0"

__all__ = [
    "CostCap",
    "CostExceededError",
    "EstimatedCost",
    "ModelPrice",
    "UnknownModelError",
    "__version__",
    "builtin_prices",
    "known_models",
]
