"""Built-in price table for common LLM models.

All rates are USD per million tokens. The numbers below reflect published
provider pricing pages as of 2026-05-24. Update locally if a provider has
changed pricing since.

Where a vendor publishes a cached / prompt-cache read rate distinct from
the standard input rate (currently Anthropic), it is exposed via
`cached_input_per_million_usd`. Models without a published cached rate
leave it as None.

Aliases: each canonical model id is registered, and common short aliases
point at the same `ModelPrice` instance so callers can pass either the
fully-qualified id or the short name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token pricing for one model.

    `cached_input_per_million_usd` covers vendor prompt-cache reads; pass
    None when the vendor does not publish a distinct cached read rate.
    """

    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: float | None = None


# Canonical models. Numbers are USD per 1M tokens as of 2026-05-24.
#
# Anthropic (https://www.anthropic.com/pricing):
#   - Opus tier:  input $15, output $75, cached read $1.50
#   - Sonnet tier: input $3, output $15, cached read $0.30
#   - Haiku tier:  input $0.80, output $4, cached read $0.08
# OpenAI (https://openai.com/api/pricing):
#   - gpt-5 family input $1.25, output $10
#   - gpt-5.4 input $1.25, output $10
# Google Gemini (https://ai.google.dev/pricing):
#   - 2.5 Pro:   input $1.25, output $10 (under 200k tokens)
#   - 2.5 Flash: input $0.30, output $2.50
# AWS Bedrock pass-through pricing tracks the vendor list above for the
# same model family; entries below are the Bedrock model ids that map to
# the same underlying Anthropic / Meta rates.
_BUILTIN: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-4-7": ModelPrice(15.00, 75.00, 1.50),
    "claude-opus-4-6": ModelPrice(15.00, 75.00, 1.50),
    "claude-opus-4-5": ModelPrice(15.00, 75.00, 1.50),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00, 0.30),
    "claude-haiku-4-5": ModelPrice(0.80, 4.00, 0.08),
    # OpenAI
    "gpt-5.4": ModelPrice(1.25, 10.00),
    "gpt-5": ModelPrice(1.25, 10.00),
    "gpt-5-mini": ModelPrice(0.25, 2.00),
    "gpt-5-nano": ModelPrice(0.05, 0.40),
    # Google Gemini
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    # AWS Bedrock - Anthropic on Bedrock matches list pricing above.
    "anthropic.claude-opus-4-7-v1:0": ModelPrice(15.00, 75.00, 1.50),
    "anthropic.claude-sonnet-4-6-v1:0": ModelPrice(3.00, 15.00, 0.30),
    "anthropic.claude-haiku-4-5-v1:0": ModelPrice(0.80, 4.00, 0.08),
    # AWS Bedrock - Meta Llama 3.1 70B (representative non-Anthropic entry)
    "meta.llama3-1-70b-instruct-v1:0": ModelPrice(2.65, 3.50),
}

# Short aliases. Each alias is the same object reference so identity
# checks remain consistent.
_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "gpt5": "gpt-5",
    "gpt-5.4": "gpt-5.4",
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
}


def builtin_prices() -> dict[str, ModelPrice]:
    """Return a fresh dict of built-in prices, with aliases resolved.

    Callers receive a copy so they can mutate or merge without affecting
    other consumers of the table.
    """
    table: dict[str, ModelPrice] = dict(_BUILTIN)
    for alias, canonical in _ALIASES.items():
        if canonical in _BUILTIN:
            table[alias] = _BUILTIN[canonical]
    return table


def known_models() -> list[str]:
    """Return the sorted list of model ids (including aliases) the built-in
    table understands."""
    return sorted({*_BUILTIN.keys(), *_ALIASES.keys()})
