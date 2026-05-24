"""Core CostCap implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from llm_cost_cap.prices import ModelPrice, builtin_prices

T = TypeVar("T")


class CostExceededError(Exception):
    """Raised when an estimated single-call cost exceeds the configured cap.

    Attributes:
        estimated_usd: total estimated cost in USD for the rejected call
        cap_usd: the configured per-call cap in USD
        model: the model id passed to ``check()`` or ``run()``
    """

    def __init__(self, estimated_usd: float, cap_usd: float, model: str):
        self.estimated_usd = estimated_usd
        self.cap_usd = cap_usd
        self.model = model
        super().__init__(
            f"estimated cost ${estimated_usd:.6f} for model {model!r} "
            f"exceeds cap ${cap_usd:.6f}"
        )


class UnknownModelError(Exception):
    """Raised when a model id is not in the price table.

    Attributes:
        model: the unknown model id
    """

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"unknown model {model!r}: pass a `prices=` override or add it "
            f"to your CostCap price table"
        )


@dataclass(frozen=True)
class EstimatedCost:
    """A breakdown of the per-call estimate for one model.

    All four fields are USD. `total_usd == input_usd + output_usd + cached_input_usd`.
    """

    total_usd: float
    input_usd: float
    output_usd: float
    cached_input_usd: float


class CostCap:
    """Pre-flight cost gate for a single LLM call.

    Build once with the per-call USD cap and (optionally) a price table
    override, then call ``check()`` before sending each request. If the
    estimated cost for the requested model + token counts exceeds the cap,
    ``CostExceededError`` is raised and the caller can short-circuit
    before paying for the call.

    Cost model:
      * ``input_tokens`` are billed at the model's input rate.
      * ``max_output_tokens`` are billed at the model's output rate. We
        use the worst-case ceiling because the cap protects against the
        worst case, not the average.
      * ``cached_input_tokens`` are billed at the model's cached-input
        rate if the model publishes one. If the model has no cached rate,
        cached tokens cost zero in the estimate (no double counting -
        they are billed nowhere else).

    Methods:
      * ``estimate(...)`` returns the breakdown without raising.
      * ``check(...)`` returns the breakdown but raises on overage.
      * ``run(model, ..., fn, *args, **kw)`` calls ``check`` and then
        invokes ``fn``. ``fn`` is not called on overage.
    """

    def __init__(
        self,
        max_usd_per_call: float,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        if max_usd_per_call < 0:
            raise ValueError("max_usd_per_call must be >= 0")
        self._cap = float(max_usd_per_call)
        # Always carry our own copy so callers can't mutate it under us.
        self._prices: dict[str, ModelPrice] = (
            dict(prices) if prices is not None else builtin_prices()
        )

    # ---- introspection ----

    @property
    def cap_usd(self) -> float:
        return self._cap

    def add_model(self, model: str, price: ModelPrice) -> None:
        """Register or replace one model in the price table."""
        self._prices[model] = price

    def known_models(self) -> list[str]:
        return sorted(self._prices.keys())

    # ---- estimation ----

    def estimate(
        self,
        *,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> EstimatedCost:
        """Return the per-call cost breakdown. Does not raise on overage.

        Raises ``UnknownModelError`` if ``model`` is not in the price table.
        Raises ``ValueError`` on negative token counts.
        """
        if input_tokens < 0 or max_output_tokens < 0 or cached_input_tokens < 0:
            raise ValueError("token counts must be non-negative")
        price = self._prices.get(model)
        if price is None:
            raise UnknownModelError(model)

        input_usd = (input_tokens / 1_000_000.0) * price.input_per_million_usd
        output_usd = (max_output_tokens / 1_000_000.0) * price.output_per_million_usd
        # Cached input is only billed when the model publishes a cached rate.
        # When no cached rate exists, we treat cached tokens as zero cost -
        # the assumption is the caller separately counted those same tokens
        # under `input_tokens` only if the vendor actually charges full rate.
        cached_input_usd = 0.0
        if cached_input_tokens > 0 and price.cached_input_per_million_usd is not None:
            cached_input_usd = (
                cached_input_tokens / 1_000_000.0
            ) * price.cached_input_per_million_usd

        total_usd = input_usd + output_usd + cached_input_usd
        return EstimatedCost(
            total_usd=total_usd,
            input_usd=input_usd,
            output_usd=output_usd,
            cached_input_usd=cached_input_usd,
        )

    # ---- gating ----

    def check(
        self,
        *,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> EstimatedCost:
        """Estimate the call and raise ``CostExceededError`` if over cap.

        Returns the ``EstimatedCost`` on success so callers can log /
        record / budget against the same number that gated the call.
        """
        estimate = self.estimate(
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        if estimate.total_usd > self._cap:
            raise CostExceededError(estimate.total_usd, self._cap, model)
        return estimate

    def run(
        self,
        *,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        fn: Callable[..., T],
        cached_input_tokens: int = 0,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> T:
        """Gate then invoke. Returns whatever ``fn`` returns.

        ``fn`` is only called if ``check`` passes. Any exception ``fn``
        raises propagates unchanged.
        """
        self.check(
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        return fn(*args, **(kwargs or {}))
