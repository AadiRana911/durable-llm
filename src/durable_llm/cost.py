"""Token pricing and exact cost arithmetic.

Every dollar figure in durable_llm originates here, and every one of them is a
:class:`decimal.Decimal`. Floats are banned from money for the usual reason —
``0.1 + 0.2 != 0.3`` — but also for a specific one: costs are summed thousands of times
into ``runs.spent_usd`` and then compared against a budget. Binary rounding error in that
accumulator does not stay small, and the whole point of a budget is that the number it
compares against is the number the invoice will show.

Results are quantised to six decimal places with ``ROUND_HALF_UP``, matching the
``numeric(12,6)`` columns in the schema, so a value computed here survives a round-trip
through Postgres unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

__all__ = ["PRICES", "Price", "cost_of", "quantize_usd", "register_price"]

logger = logging.getLogger("durable_llm.cost")

#: The scale of the ``numeric(12,6)`` money columns: one millionth of a dollar.
_QUANTUM: Final[Decimal] = Decimal("0.000001")
_TOKENS_PER_MTOK: Final[Decimal] = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    """USD per 1,000,000 tokens, input and output priced separately.

    Both fields are coerced to :class:`Decimal` on construction, going through ``str()``
    first so that ``Price(15.0, 75.0)`` cannot smuggle a binary float artefact into the
    ledger. ``Price("2", "10")`` and ``Price(Decimal("2"), Decimal("10"))`` are equivalent.
    """

    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def __post_init__(self) -> None:
        for field in ("input_per_mtok", "output_per_mtok"):
            value = getattr(self, field)
            coerced = value if isinstance(value, Decimal) else Decimal(str(value))
            if coerced < 0:
                raise ValueError(f"{field} must not be negative, got {coerced}")
            object.__setattr__(self, field, coerced)


# ======================================================================================
#  !!  DATED PRICE SNAPSHOT — VERIFY BEFORE BILLING  !!
#
#  Standard global Claude API base input/output prices below were verified on 2026-09-03
#  against https://platform.claude.com/docs/en/about-claude/pricing. They do not model
#  prompt caching, batch processing, regional/US-only inference, Fast mode, or negotiated
#  rates, and they cannot update themselves when vendor pricing changes. Before using
#  durable_llm for billing, budgeting, or chargeback, re-check that page and use
#  register_price() for a changed or workload-specific rate.
#
#  `echo` is genuinely zero: it is the offline fake provider and makes no billed calls.
# ======================================================================================
#  Model ids follow one convention: the published alias. A dated snapshot is registered
#  alongside its alias rather than instead of it, because `_price` looks the model up by
#  whatever id the provider echoed back and falls back to the requested one — so a table
#  that knows only the snapshot cannot price a call the API answered under the alias, and
#  a run that cannot be priced is a run that fails.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(Decimal("5"), Decimal("25")),
    "claude-sonnet-5": Price(Decimal("2"), Decimal("10")),
    "claude-haiku-4-5": Price(Decimal("1"), Decimal("5")),
    "claude-haiku-4-5-20251001": Price(Decimal("1"), Decimal("5")),
    "echo": Price(Decimal("0"), Decimal("0")),
}


def quantize_usd(amount: Decimal) -> Decimal:
    """Round a dollar amount to the six decimal places the schema stores.

    ``ROUND_HALF_UP`` rather than Python's default banker's rounding: it is what a finance
    person checking the arithmetic by hand will expect, and consistency with the reader
    matters more here than statistical neutrality over many roundings.
    """
    return amount.quantize(_QUANTUM, rounding=ROUND_HALF_UP)


def register_price(model: str, price: Price) -> None:
    """Add or replace the price of ``model``.

    This is the supported way to refresh the dated table above, to price a model
    durable_llm has never heard of, or to apply a negotiated rate. Overwriting is intentional —
    prices change, and a process that has to restart to pick up a new rate is a process
    that keeps billing at the old one.
    """
    if model in PRICES and PRICES[model] != price:
        logger.info("replacing price for model %s", model)
    PRICES[model] = price


def cost_of(model: str, tokens_in: int, tokens_out: int) -> Decimal:
    """Exact USD cost of one completion.

    Raises ``KeyError`` for an unknown model rather than defaulting to zero. A silent zero
    would be the worst possible failure here: the run would look free, the budget check
    would always pass, and the discrepancy would surface as an invoice.
    """
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError(f"token counts must not be negative, got in={tokens_in} out={tokens_out}")

    try:
        price = PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES)) or "<none>"
        raise KeyError(
            f"no price registered for model {model!r}; "
            "call durable_llm.cost.register_price("
            f"{model!r}, Price(input_per_mtok, output_per_mtok)) "
            f"before running it. Known models: {known}"
        ) from None

    raw = (
        Decimal(tokens_in) * price.input_per_mtok + Decimal(tokens_out) * price.output_per_mtok
    ) / _TOKENS_PER_MTOK
    return quantize_usd(raw)
