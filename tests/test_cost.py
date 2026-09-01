"""Cost arithmetic: exact, six decimal places, and never a float.

Three things are being defended here.

**Decimal, end to end.** ``spent_usd`` is an accumulator that a budget check compares
against a cap. Binary floating point error in an accumulator does not stay small, and the
whole value of a cap is that the number it compares is the number the invoice will show.
So the tests check the *type*, not just the value — a float that happens to compare equal
today is a rounding bug tomorrow.

**Rounding is ``ROUND_HALF_UP``, not Python's default.** Python and ``decimal`` both round
half to even by default, which is statistically neater and completely surprising to the
person checking your arithmetic by hand. :func:`test_rounding_is_half_up_not_half_even`
uses a value that lands exactly on the boundary, so it fails if the default is ever
restored.

**An unknown model raises.** The tempting alternative — default an unpriced model to zero —
is the worst possible behaviour: the run looks free, every budget check passes, and the
discrepancy surfaces as an invoice.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import pytest

from llmrun import cost as cost_module
from llmrun.cost import PRICES, Price, cost_of, quantize_usd, register_price

#: A model name no seeded price can collide with.
FAKE_MODEL: Final[str] = "test-only-model"


@pytest.fixture(autouse=True)
def _price_table_sandbox() -> Iterator[None]:
    """Restore ``PRICES`` after every test.

    ``PRICES`` is module-level mutable state by design (``register_price`` has to be able
    to correct a price without a process restart), which makes it exactly the kind of thing
    that leaks between tests if nobody puts it back.
    """
    snapshot = dict(PRICES)
    try:
        yield
    finally:
        PRICES.clear()
        PRICES.update(snapshot)


# --------------------------------------------------------------------------------------
# precision
# --------------------------------------------------------------------------------------


def test_cost_is_decimal_never_float() -> None:
    result = cost_of("claude-sonnet-5", 1_000_000, 0)

    assert isinstance(result, Decimal)
    assert not isinstance(result, float)
    assert result == Decimal("3.000000")


def test_result_is_quantised_to_six_decimal_places() -> None:
    """Six places is the scale of the ``numeric(12,6)`` columns.

    Matching it here means a value computed in Python survives a round-trip through
    Postgres unchanged, so ``spent_usd`` never drifts from the sum of the ledger.
    """
    result = cost_of("claude-sonnet-5", 1, 1)

    assert result.as_tuple().exponent == -6
    assert str(result) == "0.000018"  # (1 * 3 + 1 * 15) / 1e6


def test_input_and_output_tokens_are_priced_separately() -> None:
    register_price(FAKE_MODEL, Price(Decimal("2"), Decimal("10")))

    assert cost_of(FAKE_MODEL, 1_000_000, 0) == Decimal("2.000000")
    assert cost_of(FAKE_MODEL, 0, 1_000_000) == Decimal("10.000000")
    assert cost_of(FAKE_MODEL, 1_000_000, 1_000_000) == Decimal("12.000000")


def test_summing_many_small_costs_stays_exact() -> None:
    """The failure mode floats would produce, asserted against directly.

    A tenth of a cent, ten thousand times, is exactly ten dollars. In binary floating
    point it is not, and the error compounds in the direction of the budget check.
    """
    register_price(FAKE_MODEL, Price(Decimal("1000"), Decimal("0")))
    unit = cost_of(FAKE_MODEL, 1, 0)  # 1000/1e6 = 0.001 exactly

    total = sum((unit for _ in range(10_000)), Decimal(0))

    assert unit == Decimal("0.001000")
    assert total == Decimal("10.000000")


def test_rounding_is_half_up_not_half_even() -> None:
    """Half a millionth of a dollar rounds *up*.

    0.5 USD per million tokens is 0.0000005 for a single token — exactly on the six-decimal
    boundary. ``ROUND_HALF_UP`` gives 0.000001; Python's default ``ROUND_HALF_EVEN`` gives
    0.000000. This test is the tripwire on that default coming back.
    """
    register_price(FAKE_MODEL, Price(Decimal("0.5"), Decimal("0")))

    assert cost_of(FAKE_MODEL, 1, 0) == Decimal("0.000001")
    assert quantize_usd(Decimal("0.0000005")) == Decimal("0.000001")
    assert quantize_usd(Decimal("0.0000015")) == Decimal("0.000002")
    assert quantize_usd(Decimal("0.0000004")) == Decimal("0.000000")


def test_free_models_cost_nothing() -> None:
    """``echo`` is genuinely zero — it makes no billed calls — and must price as zero."""
    assert cost_of("echo", 10_000, 10_000) == Decimal("0.000000")


def test_zero_tokens_costs_nothing() -> None:
    assert cost_of("claude-opus-5", 0, 0) == Decimal("0.000000")


def test_negative_token_counts_are_rejected() -> None:
    """A negative token count is a bug upstream; pricing it would create a credit."""
    with pytest.raises(ValueError, match="negative"):
        cost_of("claude-sonnet-5", -1, 0)

    with pytest.raises(ValueError, match="negative"):
        cost_of("claude-sonnet-5", 0, -1)


# --------------------------------------------------------------------------------------
# the price table
# --------------------------------------------------------------------------------------


def test_unknown_model_raises_keyerror_naming_register_price() -> None:
    """The error has to tell you how to fix it, not just that you are wrong."""
    with pytest.raises(KeyError) as excinfo:
        cost_of("gpt-does-not-exist", 100, 100)

    message = str(excinfo.value)
    assert "gpt-does-not-exist" in message
    assert "register_price" in message


def test_register_price_round_trip() -> None:
    price = Price(Decimal("7.25"), Decimal("36.5"))

    register_price(FAKE_MODEL, price)

    assert PRICES[FAKE_MODEL] == price
    assert cost_of(FAKE_MODEL, 1_000_000, 1_000_000) == Decimal("43.750000")


def test_register_price_overwrites_an_existing_model() -> None:
    """Prices change. A process that must restart to bill correctly bills incorrectly."""
    original = cost_of("claude-sonnet-5", 1_000_000, 0)

    register_price("claude-sonnet-5", Price(Decimal("4"), Decimal("20")))

    assert original == Decimal("3.000000")
    assert cost_of("claude-sonnet-5", 1_000_000, 0) == Decimal("4.000000")


def test_prices_are_seeded_for_the_documented_models() -> None:
    """The contract names these four. A missing one is a KeyError at run time."""
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "echo"):
        assert model in PRICES, f"{model} must have a seeded price"


def test_price_fields_are_decimal_even_when_constructed_from_floats() -> None:
    """``Price(0.1, 0.2)`` must not smuggle a binary artefact into the ledger.

    ``Decimal(0.1)`` is 0.1000000000000000055511151231257827. ``Decimal(str(0.1))`` is
    exactly 0.1, which is why the coercion goes through ``str`` first.
    """
    price = Price(0.1, 0.2)  # type: ignore[arg-type]

    assert price.input_per_mtok == Decimal("0.1")
    assert price.output_per_mtok == Decimal("0.2")
    assert isinstance(price.input_per_mtok, Decimal)


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        Price(Decimal("-1"), Decimal("1"))


def test_price_table_is_documented_as_a_placeholder() -> None:
    """The seeded figures are illustrative, and the module must say so.

    Presenting unverified vendor pricing as authoritative is how a cost tool ends up
    quoted in someone's budget spreadsheet. If the warning above ``PRICES`` is ever
    deleted, this fails.
    """
    source = inspect.getsource(cost_module)

    assert "PLACEHOLDER" in source.upper()
