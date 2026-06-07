"""Tests for the CostCap gate."""

from __future__ import annotations

import pytest

from llm_cost_cap import (
    CostCap,
    CostExceededError,
    EstimatedCost,
    ModelPrice,
    UnknownModelError,
)

# ---------- estimate ----------


def test_estimate_matches_hand_calculation_for_opus():
    cap = CostCap(max_usd_per_call=10.0)
    est = cap.estimate(
        model="claude-opus-4-7",
        input_tokens=10_000,
        max_output_tokens=4_000,
    )
    # opus is $15/M input, $75/M output
    # input  = 10_000 * 15 / 1_000_000 = 0.15
    # output = 4_000 * 75 / 1_000_000  = 0.30
    expected_input = 0.15
    expected_output = 0.30
    assert est.input_usd == pytest.approx(expected_input)
    assert est.output_usd == pytest.approx(expected_output)
    assert est.cached_input_usd == 0.0
    assert est.total_usd == pytest.approx(expected_input + expected_output)


def test_estimate_for_haiku_is_cheaper_than_opus():
    cap = CostCap(max_usd_per_call=10.0)
    opus = cap.estimate(model="claude-opus-4-7", input_tokens=1000, max_output_tokens=1000)
    haiku = cap.estimate(model="claude-haiku-4-5", input_tokens=1000, max_output_tokens=1000)
    assert haiku.total_usd < opus.total_usd


def test_estimate_returns_breakdown_fields():
    cap = CostCap(max_usd_per_call=10.0)
    est = cap.estimate(model="gpt-5", input_tokens=1_000_000, max_output_tokens=100_000)
    # gpt-5 input $1.25, output $10
    assert isinstance(est, EstimatedCost)
    assert est.input_usd == pytest.approx(1.25)
    assert est.output_usd == pytest.approx(1.0)
    assert est.total_usd == pytest.approx(2.25)


def test_estimate_does_not_raise_even_when_over_cap():
    cap = CostCap(max_usd_per_call=0.0001)
    est = cap.estimate(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        max_output_tokens=1_000_000,
    )
    # estimate alone does NOT raise even though we're massively over cap
    assert est.total_usd > cap.cap_usd


def test_estimate_rejects_negative_token_counts():
    cap = CostCap(max_usd_per_call=1.0)
    with pytest.raises(ValueError):
        cap.estimate(model="gpt-5", input_tokens=-1, max_output_tokens=10)
    with pytest.raises(ValueError):
        cap.estimate(model="gpt-5", input_tokens=10, max_output_tokens=-1)
    with pytest.raises(ValueError):
        cap.estimate(
            model="gpt-5",
            input_tokens=10,
            max_output_tokens=10,
            cached_input_tokens=-1,
        )


# ---------- cached input ----------


def test_cached_input_reduces_cost_when_model_publishes_cached_rate():
    cap = CostCap(max_usd_per_call=10.0)
    # Anthropic sonnet publishes cached input at $0.30/M (vs $3/M input)
    est = cap.estimate(
        model="claude-sonnet-4-6",
        input_tokens=0,
        max_output_tokens=0,
        cached_input_tokens=1_000_000,
    )
    assert est.cached_input_usd == pytest.approx(0.30)
    assert est.total_usd == pytest.approx(0.30)


def test_cached_input_is_zero_when_model_has_no_cached_rate():
    cap = CostCap(max_usd_per_call=10.0)
    # gpt-5 has no published cached rate in our table
    est = cap.estimate(
        model="gpt-5",
        input_tokens=0,
        max_output_tokens=0,
        cached_input_tokens=1_000_000,
    )
    assert est.cached_input_usd == 0.0


def test_total_equals_sum_of_all_three_components():
    # Documented invariant: total_usd == input_usd + output_usd + cached_input_usd
    cap = CostCap(max_usd_per_call=10.0)
    est = cap.estimate(
        model="claude-sonnet-4-6",
        input_tokens=2_000,
        max_output_tokens=500,
        cached_input_tokens=1_500,
    )
    assert est.input_usd > 0
    assert est.output_usd > 0
    assert est.cached_input_usd > 0
    assert est.total_usd == pytest.approx(est.input_usd + est.output_usd + est.cached_input_usd)


# ---------- check ----------


def test_check_passes_when_under_cap():
    cap = CostCap(max_usd_per_call=0.50)
    est = cap.check(model="claude-opus-4-7", input_tokens=1000, max_output_tokens=1000)
    # 1000 input * 15/M + 1000 output * 75/M = 0.015 + 0.075 = 0.090
    assert est.total_usd == pytest.approx(0.090)
    assert est.total_usd < 0.50


def test_check_raises_when_over_cap_with_correct_attrs():
    cap = CostCap(max_usd_per_call=0.10)
    with pytest.raises(CostExceededError) as excinfo:
        cap.check(
            model="claude-opus-4-7",
            input_tokens=10_000,
            max_output_tokens=4_000,
        )
    err = excinfo.value
    assert err.model == "claude-opus-4-7"
    assert err.cap_usd == 0.10
    # estimated is 0.45 from earlier hand-calc
    assert err.estimated_usd == pytest.approx(0.45)
    assert err.estimated_usd > err.cap_usd
    # the message references both numbers
    assert "0.45" in str(err) or "0.450000" in str(err)


def test_check_equal_to_cap_passes_not_raises():
    cap = CostCap(max_usd_per_call=10.0)
    # Construct a custom model where the estimate is exactly 1.0
    custom = ModelPrice(input_per_million_usd=1_000_000.0, output_per_million_usd=0.0)
    cap.add_model("tester", custom)
    est = cap.check(model="tester", input_tokens=1, max_output_tokens=0)
    assert est.total_usd == pytest.approx(1.0)


# ---------- run ----------


def test_run_invokes_fn_and_returns_result_when_under_cap():
    cap = CostCap(max_usd_per_call=10.0)
    calls: list[str] = []

    def fake_llm_call(prompt: str) -> str:
        calls.append(prompt)
        return f"echo: {prompt}"

    result = cap.run(
        model="claude-sonnet-4-6",
        input_tokens=500,
        max_output_tokens=500,
        fn=fake_llm_call,
        args=("hello",),
    )
    assert result == "echo: hello"
    assert calls == ["hello"]


def test_run_propagates_fn_exception_when_under_cap():
    cap = CostCap(max_usd_per_call=10.0)

    def boom():
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        cap.run(
            model="gpt-5",
            input_tokens=10,
            max_output_tokens=10,
            fn=boom,
        )


def test_run_does_not_invoke_fn_when_over_cap():
    cap = CostCap(max_usd_per_call=0.0001)
    calls: list[int] = []

    def fake_llm_call() -> str:
        calls.append(1)
        return "should not happen"

    with pytest.raises(CostExceededError):
        cap.run(
            model="claude-opus-4-7",
            input_tokens=10_000,
            max_output_tokens=4_000,
            fn=fake_llm_call,
        )
    assert calls == [], "fn must NOT be called when cap is exceeded"


def test_run_applies_cached_input_in_gate():
    cap = CostCap(max_usd_per_call=10.0)
    calls: list[int] = []
    out = cap.run(
        model="claude-opus-4-7",
        input_tokens=1_000,
        max_output_tokens=1_000,
        cached_input_tokens=500,
        fn=lambda: calls.append(1) or "ok",
    )
    assert out == "ok"
    assert calls == [1]


def test_run_passes_kwargs_to_fn():
    cap = CostCap(max_usd_per_call=10.0)

    def fake(prefix: str, suffix: str) -> str:
        return f"{prefix}:{suffix}"

    out = cap.run(
        model="gpt-5",
        input_tokens=10,
        max_output_tokens=10,
        fn=fake,
        args=("hi",),
        kwargs={"suffix": "world"},
    )
    assert out == "hi:world"


# ---------- price-table behaviour ----------


def test_unknown_model_raises_unknown_model_error():
    cap = CostCap(max_usd_per_call=1.0)
    with pytest.raises(UnknownModelError) as excinfo:
        cap.check(
            model="some-random-model",
            input_tokens=10,
            max_output_tokens=10,
        )
    assert excinfo.value.model == "some-random-model"
    # message hints at how to fix
    assert "some-random-model" in str(excinfo.value)


def test_custom_prices_override_builtin_table_entirely():
    custom_prices = {
        "my-private-model": ModelPrice(2.0, 8.0),
    }
    cap = CostCap(max_usd_per_call=10.0, prices=custom_prices)

    # built-in model is no longer known when an override table is provided
    with pytest.raises(UnknownModelError):
        cap.check(
            model="claude-opus-4-7",
            input_tokens=10,
            max_output_tokens=10,
        )

    # the custom model is gated against the new prices
    est = cap.check(
        model="my-private-model",
        input_tokens=1_000_000,
        max_output_tokens=100_000,
    )
    # 1_000_000 * 2/M + 100_000 * 8/M = 2.0 + 0.8 = 2.8
    assert est.input_usd == pytest.approx(2.0)
    assert est.output_usd == pytest.approx(0.8)
    assert est.total_usd == pytest.approx(2.8)


def test_add_model_registers_one_model_into_an_existing_cap():
    cap = CostCap(max_usd_per_call=10.0)
    cap.add_model("homemade", ModelPrice(0.5, 1.5))
    est = cap.check(model="homemade", input_tokens=1_000_000, max_output_tokens=1_000_000)
    assert est.input_usd == pytest.approx(0.5)
    assert est.output_usd == pytest.approx(1.5)


def test_caller_cannot_mutate_internal_price_table_via_input_dict():
    custom: dict[str, ModelPrice] = {"x": ModelPrice(1.0, 1.0)}
    cap = CostCap(max_usd_per_call=10.0, prices=custom)
    # caller now adds another entry to their dict
    custom["y"] = ModelPrice(2.0, 2.0)
    # cap was constructed with a copy, so "y" is not visible
    with pytest.raises(UnknownModelError):
        cap.check(model="y", input_tokens=10, max_output_tokens=10)


def test_known_models_lists_registered_entries():
    cap = CostCap(max_usd_per_call=1.0, prices={"a": ModelPrice(1, 2), "b": ModelPrice(3, 4)})
    assert cap.known_models() == ["a", "b"]


def test_default_cap_known_models_includes_builtin_aliases():
    cap = CostCap(max_usd_per_call=1.0)
    models = cap.known_models()
    # canonical id and its short alias are both gateable on a default cap
    assert "claude-opus-4-7" in models
    assert "opus" in models
    # gating via the alias resolves to the same price as the canonical id
    via_alias = cap.estimate(model="opus", input_tokens=1000, max_output_tokens=1000)
    via_canonical = cap.estimate(model="claude-opus-4-7", input_tokens=1000, max_output_tokens=1000)
    assert via_alias.total_usd == via_canonical.total_usd


# ---------- constructor + property ----------


def test_negative_cap_raises_at_construction():
    with pytest.raises(ValueError):
        CostCap(max_usd_per_call=-0.01)


def test_cap_usd_property_returns_configured_value():
    cap = CostCap(max_usd_per_call=2.5)
    assert cap.cap_usd == 2.5
