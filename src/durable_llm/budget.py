"""Pre-flight budget enforcement.

Enforcement happens **before** the network call, never after. This is the whole design:

    A cap you discover you have exceeded is not a cap. It is an invoice.

Checking spend afterwards can only ever report the overrun — the tokens are already
billed. So every call site projects its worst-case cost first, asks whether that cost
still fits under the run's cap, and refuses to make the call if it does not. The
projection is deliberately pessimistic on the *output* side (assume the model emits every
one of ``max_tokens``), and that half is genuinely bounded.

The input half is an estimate and can be wrong in the expensive direction, which is why
:func:`budget_overrun` exists. ``prompt_chars / 4`` is roughly right for English prose and
too low for CJK, Cyrillic, Devanagari, base64 and minified JSON, where a real tokeniser
emits far more than one token per four characters; a provider can also report a pricier
model than the one that was projected. Left there, the cap would silently not hold. So the
caller re-checks the *actual* cost against the cap after the call, in the same transaction
as the ledger row, and halts the run before the next call. The residual exposure is one
call wide — the same width as the crash window, and for the same reason: you cannot
un-spend money, you can only refuse to spend more.

Concurrency matters here too. Two workers checking the same run at the same moment would
both read the same ``spent_usd``, both conclude there is room, and both spend it. So
:func:`check_budget` takes a row lock with ``SELECT ... FOR UPDATE`` and — crucially —
does not commit. The lock is held by the *caller's* transaction until the caller commits
the ``llm_calls`` row and the updated spend, which serialises budget checks for a run
without serialising anything else.

The helpers here expect a connection whose cursors yield mappings, i.e. the ``dict_row``
factory configured by ``durable_llm.db.connect()``.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Final
from uuid import UUID

from durable_llm.cost import cost_of, quantize_usd
from durable_llm.errors import BudgetExceededError, PermanentError
from durable_llm.types import as_decimal, as_decimal_or_zero

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module importable bare
    import psycopg
    from psycopg.rows import DictRow

__all__ = ["budget_overrun", "check_budget", "project_cost", "record_spend"]

logger = logging.getLogger("durable_llm.budget")

#: Rule-of-thumb characters per token used to project input size without tokenising.
#: Roughly right for English prose and cheap to compute; the projection is a pre-flight
#: guard, not an accounting figure, and the real token counts come back from the provider.
#:
#: It is an *under*-estimate for anything that is not English prose — CJK and Cyrillic text,
#: base64, minified JSON, source code. Lowering it would make the guard stricter at the cost
#: of refusing calls that would have fit; :func:`budget_overrun` closes the gap instead, by
#: comparing what the call actually cost against the cap once the provider has answered.
CHARS_PER_TOKEN: Final[int] = 4

# The ``::uuid`` casts are on the *parameter*, not the column, so the primary-key index is
# still used. They let the helpers accept either a UUID object or its string form without
# tripping over "operator does not exist: uuid = text".
_SELECT_BUDGET_FOR_UPDATE: Final[str] = """
    SELECT budget_usd, spent_usd
    FROM runs
    WHERE id = %s::uuid
    FOR UPDATE
"""

# No FOR UPDATE: the caller already holds this row's lock (it just wrote the spend), and
# re-locking would only add a second lock acquisition to the hot path for nothing.
_SELECT_BUDGET: Final[str] = """
    SELECT budget_usd, spent_usd
    FROM runs
    WHERE id = %s::uuid
"""

_UPDATE_SPEND: Final[str] = """
    UPDATE runs
    SET spent_usd = spent_usd + %s,
        updated_at = now()
    WHERE id = %s::uuid
"""


def project_cost(model: str, prompt_chars: int, max_tokens: int) -> Decimal:
    """Worst-case USD cost of a call that has not happened yet.

    Input tokens are estimated as ``ceil(prompt_chars / 4)`` and the model is assumed to
    emit the full ``max_tokens`` of output. Both assumptions round against the caller on
    purpose: a budget guard that under-projects is a budget guard that lets the overrun
    through.

    Raises ``KeyError`` (from :func:`durable_llm.cost.cost_of`) for an unpriced model, so an
    unknown model is caught before the call rather than after it.
    """
    if prompt_chars < 0:
        raise ValueError(f"prompt_chars must not be negative, got {prompt_chars}")
    if max_tokens < 0:
        raise ValueError(f"max_tokens must not be negative, got {max_tokens}")

    # Integer ceiling division: exact for arbitrarily large inputs, unlike math.ceil on a
    # float quotient.
    estimated_tokens_in = (prompt_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
    return cost_of(model, estimated_tokens_in, max_tokens)


def check_budget(conn: psycopg.Connection[DictRow], run_id: UUID, projected: Decimal) -> None:
    """Refuse the call if ``projected`` would push the run past its cap.

    Locks the run row for the duration of the caller's transaction and raises
    :class:`~durable_llm.errors.BudgetExceededError` when ``spent + projected > budget``. A
    ``NULL`` budget means unlimited.

    The comparison is strictly greater-than, so a projection that lands exactly on the cap
    is allowed: the cap is the amount you may spend, not the amount you must stay below.

    **Does not commit and does not roll back.** It runs inside whatever transaction the
    caller has open, which is what keeps the ``FOR UPDATE`` lock alive until the resulting
    spend has been durably recorded. Committing here would release the lock while the
    provider call is still in flight and reopen the race it exists to close.
    """
    with conn.cursor() as cur:
        cur.execute(_SELECT_BUDGET_FOR_UPDATE, (str(run_id),))
        row = cur.fetchone()

    if row is None:
        # The run vanished mid-flight, or the caller passed an id that never existed.
        # Either way it is a bug, not something to retry against a paid API.
        raise PermanentError(f"run {run_id} does not exist; cannot check its budget")

    budget = as_decimal(row["budget_usd"])
    # spent_usd is NOT NULL DEFAULT 0; the fallback only matters for hand-built fakes.
    spent = as_decimal_or_zero(row["spent_usd"])
    # Callers are expected to hand over a Decimal; coercing anyway means a stray float
    # cannot blow up the comparison in the middle of a paid pipeline.
    projected = as_decimal_or_zero(projected)

    if budget is None:
        logger.debug("run %s has no budget cap; allowing projected %s USD", run_id, projected)
        return

    if spent + projected > budget:
        raise BudgetExceededError(
            f"run {run_id} would exceed its budget: "
            f"spent {spent} + projected {projected} > budget {budget} USD"
        )

    logger.debug(
        "run %s budget ok: spent %s + projected %s <= budget %s USD",
        run_id,
        spent,
        projected,
        budget,
    )


def budget_overrun(conn: psycopg.Connection[DictRow], run_id: UUID) -> Decimal | None:
    """How far past its cap the run has actually gone, or ``None`` if it has not.

    The post-flight half of the guard. :func:`check_budget` refuses calls whose *projected*
    cost would not fit; this reports whether the *recorded* spend fits, which is the only
    number an invoice will ever agree with. Call it in the same transaction as the ledger
    row and the spend it is checking, so it reads the total it is asking about.

    Returns ``None`` for an unlimited budget, for a run that no longer exists (there is
    nothing left to halt), and for spend that is still within the cap. Deliberately does not
    raise: the caller has just written a ledger row it must commit, and an exception here
    would roll back the receipt for money that was already spent.
    """
    with conn.cursor() as cur:
        cur.execute(_SELECT_BUDGET, (str(run_id),))
        row = cur.fetchone()

    if row is None:
        return None

    budget = as_decimal(row["budget_usd"])
    if budget is None:
        return None

    spent = as_decimal_or_zero(row["spent_usd"])
    if spent <= budget:
        return None

    over = spent - budget
    logger.warning(
        "run %s has spent %s USD against a %s USD cap (%s over); the projection "
        "under-estimated the call that crossed it",
        run_id,
        spent,
        budget,
        over,
    )
    return over


def record_spend(conn: psycopg.Connection[DictRow], run_id: UUID, amount: Decimal) -> None:
    """Add ``amount`` to the run's running total.

    Written as ``spent_usd = spent_usd + %s`` rather than read-modify-write so the addition
    happens in the database and cannot lose a concurrent update.

    Like :func:`check_budget`, this does not commit: the caller must record the spend in
    the same transaction as the ``llm_calls`` row that justifies it. Splitting them lets a
    crash land between the two and produce a ledger that disagrees with the total.
    """
    amount = as_decimal_or_zero(amount)
    if amount < 0:
        # There is no such thing as a refund in a cost ledger: a provider call that was
        # billed stays billed. A negative spend means a bug upstream.
        raise ValueError(f"spend must not be negative, got {amount}")

    with conn.cursor() as cur:
        cur.execute(_UPDATE_SPEND, (quantize_usd(amount), str(run_id)))
        if cur.rowcount == 0:
            raise PermanentError(f"run {run_id} does not exist; cannot record spend")

    logger.debug("recorded %s USD against run %s", amount, run_id)
