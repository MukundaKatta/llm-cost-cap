"""Tests for the built-in price table."""

from __future__ import annotations

from llm_cost_cap import ModelPrice, builtin_prices, known_models


def test_builtin_prices_contains_required_models():
    table = builtin_prices()
    for model in (
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4",
        "gpt-5",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ):
        assert model in table, f"missing required model {model!r}"


def test_anthropic_models_have_cached_input_rate():
    table = builtin_prices()
    for model in (
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        price = table[model]
        assert price.cached_input_per_million_usd is not None, (
            f"{model} should publish a cached read rate"
        )
        # cached should always be cheaper than full input
        assert price.cached_input_per_million_usd < price.input_per_million_usd


def test_short_aliases_resolve_to_canonical_models():
    table = builtin_prices()
    # the alias and the canonical id point at the same price object
    assert table["opus"] is table["claude-opus-4-7"]
    assert table["sonnet"] is table["claude-sonnet-4-6"]
    assert table["haiku"] is table["claude-haiku-4-5"]


def test_builtin_prices_returns_a_fresh_dict_each_call():
    a = builtin_prices()
    b = builtin_prices()
    # different containers
    assert a is not b
    # mutating one does not affect the other
    a["mutated"] = ModelPrice(99.0, 99.0)
    assert "mutated" not in b


def test_model_price_is_frozen_dataclass():
    p = ModelPrice(1.0, 2.0, 0.1)
    try:
        p.input_per_million_usd = 999.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ModelPrice should be frozen")


def test_known_models_includes_aliases_and_canonical():
    models = known_models()
    assert "claude-opus-4-7" in models
    assert "opus" in models
