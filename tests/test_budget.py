"""Budget enforcement — specifically, that it happens *before* the money is spent.

A cap you discover you have exceeded is not a cap, it is an invoice. So the headline test
here is not "the run ends in ``halted_budget``" — it is that the provider recorded **zero
calls** when it did. Those are very different systems and only one of them saves money: a
post-hoc check produces exactly the same run status, having already paid for the tokens.

The rest pins the edges where budget code traditionally goes wrong:

* the exact boundary, where the projection lands precisely on the remaining budget. The
  contract says raise on ``spent + projected > budget``, strictly — the cap is the amount
  you may spend, not the amount you must stay under. An off-by-one here shows up in
  production as "the last call of every run mysteriously fails";
* ``NULL`` meaning unlimited rather than zero. Getting that backwards halts every run that
  never asked for a cap;
* worst-case projection: input tokens estimated from characters, output assumed to be the
  full ``max_tokens``. Projecting the *likely* cost instead of the worst case means the cap
  silently does not hold.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest

from durable_llm.budget import CHARS_PER_TOKEN, check_budget, project_cost, record_spend
from durable_llm.context import RunContext
from durable_llm.cost import cost_of
from durable_llm.errors import BudgetExceededError, PermanentError
from durable_llm.pipeline import pipeline
from durable_llm.providers.echo import EchoProvider
from durable_llm.types import LLMResult
from durable_llm.worker import run_once

if TYPE_CHECKING:
    from conftest import Database

#: A model with a non-zero price, so the offline echo provider can accrue real spend
#: without touching a network. ``echo`` itself is priced at zero, which makes it useless
#: for budget tests: no cap can ever be crossed by a free call.
PRICED_MODEL = "claude-sonnet-5"

MAX_TOKENS = 256

WORKER_ID = "test-budget"


def _prompt(step: str) -> str:
    """Prompt text carrying the step name — see the note in examples/summarise_thread.py."""
    return f"[{step}] Summarise the payload as a single JSON object."


def _drive(db: Database, provider: EchoProvider) -> UUID | None:
    """Claim and execute one run, exactly as ``durable-llm worker --once`` does."""
    return run_once(db.conn, worker_id=WORKER_ID, provider=provider, lease_seconds=30)


@pipeline(name="test_budget_capped", budget_usd="0.000001", max_attempts=5)
def capped_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A run whose cap is one millionth of a dollar: no priced call can fit under it."""
    answer = ctx.llm(
        "answer",
        _prompt("answer"),
        {"subject": subject},
        model=PRICED_MODEL,
        max_tokens=MAX_TOKENS,
    )
    return {"answer": answer}


@pipeline(name="test_budget_unlimited", budget_usd=None, max_attempts=5)
def unlimited_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """The same work with no cap at all — ``budget_usd`` NULL means unlimited."""
    answer = ctx.llm(
        "answer",
        _prompt("answer"),
        {"subject": subject},
        model=PRICED_MODEL,
        max_tokens=MAX_TOKENS,
    )
    return {"answer": answer}


# --------------------------------------------------------------------------------------
# the headline: enforcement happens before the call
# --------------------------------------------------------------------------------------


def test_budget_halts_before_the_provider_is_called(db: Database) -> None:
    """The whole design in one assertion: the provider recorded no calls.

    If enforcement ever moved to after the call — a one-line refactor someone will
    eventually propose — every other assertion in this test would still pass. This one
    would not.
    """
    run_id = capped_pipeline.enqueue(tenant_id="budget", conn=db.conn, subject="quarterly costs")
    db.conn.commit()
    provider = EchoProvider()

    assert _drive(db, provider) == run_id

    assert sum(provider.call_counts.values()) == 0, (
        "the provider was called before the budget was checked; those tokens are already "
        f"billed and the cap saved nothing (calls: {dict(provider.call_counts)})"
    )
    record = db.run(run_id)
    assert record.status == "halted_budget"
    assert record.error_class == "budget"
    assert record.spent_usd == Decimal("0")
    assert db.count_llm_calls(run_id) == 0


def test_a_halted_run_is_not_retried(db: Database) -> None:
    """``halted_budget`` is terminal. Retrying is the exact behaviour the cap prevents."""
    run_id = capped_pipeline.enqueue(tenant_id="budget", conn=db.conn, subject="quarterly costs")
    db.conn.commit()

    _drive(db, EchoProvider())
    second = _drive(db, EchoProvider())

    assert second is None, "a budget-halted run must not be claimed again"
    assert db.run(run_id).status == "halted_budget"


def test_unlimited_budget_runs_to_completion_and_records_the_spend(db: Database) -> None:
    """A NULL cap means *unlimited*, not zero — and the spend is still recorded."""
    run_id = unlimited_pipeline.enqueue(tenant_id="budget", conn=db.conn, subject="anything")
    db.conn.commit()
    provider = EchoProvider()

    assert _drive(db, provider) == run_id

    record = db.run(run_id)
    assert record.status == "succeeded"
    assert record.budget_usd is None
    assert sum(provider.call_counts.values()) == 1
    assert db.count_llm_calls(run_id, cached=False) == 1
    # The call was real and the model is priced, so the ledger must show money moving.
    assert isinstance(record.spent_usd, Decimal)
    assert record.spent_usd > Decimal("0")


# --------------------------------------------------------------------------------------
# the post-flight half: when the projection was wrong in the expensive direction
# --------------------------------------------------------------------------------------


class _UnderProjectedProvider:
    """Reports far more input tokens than ``prompt_chars / 4`` predicted.

    Not an exotic failure: this is simply what a real tokeniser does to CJK, Cyrillic,
    Devanagari, base64, minified JSON and source code, where one character is routinely one
    token. The pre-flight check can only ever see the projection, so this is the shape of
    every case in which the cap would silently fail to hold.
    """

    name = "under-projected"
    model = PRICED_MODEL

    def __init__(self, *, tokens_in: int) -> None:
        self._tokens_in = tokens_in
        self.calls = 0

    def complete(
        self,
        *,
        model: str = "",
        prompt: str = "",
        payload: str = "",
        max_tokens: int = 1024,
        **params: Any,
    ) -> LLMResult:
        del prompt, payload, max_tokens, params
        self.calls += 1
        return LLMResult(
            text='"answered"',
            model=model or self.model,
            tokens_in=self._tokens_in,
            tokens_out=10,
        )


#: Chosen so the *projection* for step one fits under the cap and the *actual* cost does
#: not: ~21 estimated input tokens plus 256 assumed output tokens is $0.003903 against a
#: $0.004 cap, while 4000 real input tokens is $0.012150.
_UNDER_PROJECTED_TOKENS_IN = 4000
_UNDER_PROJECTED_BUDGET = Decimal("0.004000")
_UNDER_PROJECTED_ACTUAL = Decimal("0.012150")


@pipeline(name="test_budget_under_projected_once", budget_usd="0.004", max_attempts=5)
def under_projected_once_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """Exactly one call. Nothing follows it, so only the post-flight check can catch it."""
    return {
        "first": ctx.llm(
            "first",
            _prompt("first"),
            {"subject": subject},
            model=PRICED_MODEL,
            max_tokens=MAX_TOKENS,
        )
    }


@pipeline(name="test_budget_under_projected_twice", budget_usd="0.004", max_attempts=5)
def under_projected_twice_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """Two calls, so "the exposure is one call wide" is observable rather than asserted."""
    first = ctx.llm(
        "first", _prompt("first"), {"subject": subject}, model=PRICED_MODEL, max_tokens=MAX_TOKENS
    )
    second = ctx.llm(
        "second", _prompt("second"), {"subject": subject}, model=PRICED_MODEL, max_tokens=MAX_TOKENS
    )
    return {"first": first, "second": second}


def test_a_run_whose_only_call_crosses_the_cap_halts_instead_of_succeeding(
    db: Database,
) -> None:
    """The discriminating test for the post-flight guard.

    One call, and it costs three times the cap because the projection under-counted its
    input. ``check_budget`` cannot catch this — it compared a projection that genuinely
    fit — and there is no *next* call whose pre-flight check would catch it either. Only
    the comparison of the recorded cost against the cap, in the same transaction as the
    ledger row, is left.

    Delete ``budget_overrun`` from ``_billed_call`` and this run reports ``succeeded`` with
    ``spent_usd`` at triple its cap, no error class, and nothing anywhere saying the cap
    failed to hold. That is the failure this test exists to make loud.
    """
    run_id = under_projected_once_pipeline.enqueue(
        tenant_id="budget", conn=db.conn, subject="quarterly costs"
    )
    db.conn.commit()
    provider = _UnderProjectedProvider(tokens_in=_UNDER_PROJECTED_TOKENS_IN)

    assert run_once(db.conn, worker_id=WORKER_ID, provider=provider, lease_seconds=30) == run_id

    record = db.run(run_id)
    assert record.status == "halted_budget", (
        f"the run finished {record.status!r} having spent {record.spent_usd} against a "
        f"{_UNDER_PROJECTED_BUDGET} cap; the cap silently did not hold"
    )
    assert record.error_class == "budget"

    # The receipt survived the halt. A cap that rolls back the record of the call that
    # crossed it is worse than one that reports the crossing a call late.
    assert record.spent_usd == _UNDER_PROJECTED_ACTUAL
    assert record.spent_usd > _UNDER_PROJECTED_BUDGET
    assert db.count_llm_calls(run_id, cached=False) == 1


def test_the_overrun_is_bounded_to_the_one_call_that_crossed_the_cap(db: Database) -> None:
    """The other half of the honest guarantee: one call over, not one pipeline over."""
    run_id = under_projected_twice_pipeline.enqueue(
        tenant_id="budget", conn=db.conn, subject="quarterly costs"
    )
    db.conn.commit()
    provider = _UnderProjectedProvider(tokens_in=_UNDER_PROJECTED_TOKENS_IN)

    assert run_once(db.conn, worker_id=WORKER_ID, provider=provider, lease_seconds=30) == run_id

    assert provider.calls == 1, (
        f"the run kept spending after crossing its cap: {provider.calls} provider call(s) "
        "were made where the guard should have halted after the first"
    )
    assert db.run(run_id).status == "halted_budget"
    assert db.count_llm_calls(run_id, cached=False) == 1

    # The halt is attributed to the call that actually crossed the cap, not to the next
    # one. Without the post-flight check `first` reads `succeeded` and `second` carries the
    # failure — the same terminal status, blaming the wrong step.
    steps = {step.name: step for step in db.steps(run_id)}
    assert steps["first"].status == "failed"
    assert steps["first"].error_class == "budget"
    assert "second" not in steps, "the run made it to the next step despite being over cap"


# --------------------------------------------------------------------------------------
# check_budget: the boundary
# --------------------------------------------------------------------------------------


def test_projection_exactly_equal_to_the_remaining_budget_is_allowed(db: Database) -> None:
    """Spending the last cent of a cap is spending the cap, not exceeding it.

    ``check_budget`` raises on ``spent + projected > budget``, strictly greater. If that
    ever becomes ``>=`` this test is what tells you; otherwise the symptom is the far more
    confusing "runs halt just short of their budget".
    """
    run_id = db.insert_run(budget_usd=Decimal("1.000000"), spent_usd=Decimal("0.250000"))

    check_budget(db.conn, run_id, Decimal("0.750000"))  # must not raise

    # check_budget deliberately does not commit: it holds a FOR UPDATE lock for its
    # caller's transaction. Release it, since this test is that caller.
    db.conn.rollback()


def test_one_millionth_over_the_boundary_is_refused(db: Database) -> None:
    """The other side of the same boundary, at the resolution the schema stores."""
    run_id = db.insert_run(budget_usd=Decimal("1.000000"), spent_usd=Decimal("0.250000"))

    with pytest.raises(BudgetExceededError) as excinfo:
        check_budget(db.conn, run_id, Decimal("0.750001"))
    db.conn.rollback()

    assert str(run_id) in str(excinfo.value)


def test_unlimited_budget_never_halts(db: Database) -> None:
    """A NULL ``budget_usd`` lets an absurd projection through rather than halting."""
    run_id = db.insert_run(budget_usd=None, spent_usd=Decimal("0"))

    check_budget(db.conn, run_id, Decimal("1000000.000000"))  # must not raise

    db.conn.rollback()


def test_already_spent_counts_against_the_cap(db: Database) -> None:
    """The comparison is ``spent + projected``, not the projection on its own."""
    run_id = db.insert_run(budget_usd=Decimal("1.000000"), spent_usd=Decimal("0.990000"))

    with pytest.raises(BudgetExceededError):
        check_budget(db.conn, run_id, Decimal("0.020000"))
    db.conn.rollback()


def test_zero_budget_refuses_any_non_zero_projection(db: Database) -> None:
    """Zero is a real cap, and it must stay distinguishable from NULL."""
    run_id = db.insert_run(budget_usd=Decimal("0"), spent_usd=Decimal("0"))

    check_budget(db.conn, run_id, Decimal("0"))  # a free call still fits
    with pytest.raises(BudgetExceededError):
        check_budget(db.conn, run_id, Decimal("0.000001"))
    db.conn.rollback()


def test_checking_a_missing_run_is_a_bug_not_a_retry(db: Database) -> None:
    """A vanished run is a programming error.

    Classified as transient it would be retried against a paid API; ``PermanentError`` stops
    it at the first attempt.
    """
    with pytest.raises(PermanentError):
        check_budget(db.conn, uuid4(), Decimal("0.01"))
    db.conn.rollback()


# --------------------------------------------------------------------------------------
# project_cost and record_spend
# --------------------------------------------------------------------------------------


def test_projection_assumes_the_worst_case() -> None:
    """Input estimated from characters, output assumed to be the full ``max_tokens``.

    Both assumptions round against the caller. Wrong in the cheap direction, an occasional
    call that would have fitted is refused; wrong in the other, the cap does not hold —
    and only one of those two mistakes costs money.
    """
    prompt_chars = 4_001
    expected_tokens_in = -(-prompt_chars // CHARS_PER_TOKEN)  # integer ceiling, exactly

    projected = project_cost(PRICED_MODEL, prompt_chars, 1_000)

    assert expected_tokens_in == 1_001
    assert projected == cost_of(PRICED_MODEL, expected_tokens_in, 1_000)
    assert isinstance(projected, Decimal)


def test_projection_of_an_unpriced_model_fails_before_the_call() -> None:
    """An unknown model is caught by the budget guard, not by the invoice."""
    with pytest.raises(KeyError, match="register_price"):
        project_cost("model-we-have-never-heard-of", 100, 100)


def test_record_spend_accumulates_in_the_database(db: Database) -> None:
    """Written as ``spent_usd = spent_usd + %s`` so a concurrent update cannot be lost."""
    run_id = db.insert_run(budget_usd=Decimal("5"), spent_usd=Decimal("0"))

    record_spend(db.conn, run_id, Decimal("0.100000"))
    record_spend(db.conn, run_id, Decimal("0.023456"))
    db.conn.commit()

    spent = db.spent(run_id)
    assert spent == Decimal("0.123456")
    assert isinstance(spent, Decimal)


def test_recorded_spend_is_visible_to_the_next_budget_check(db: Database) -> None:
    """The two halves have to agree, or a cap can be walked past one call at a time."""
    run_id = db.insert_run(budget_usd=Decimal("1.000000"), spent_usd=Decimal("0"))

    record_spend(db.conn, run_id, Decimal("0.900000"))
    db.conn.commit()

    check_budget(db.conn, run_id, Decimal("0.100000"))  # exactly the remainder: allowed
    with pytest.raises(BudgetExceededError):
        check_budget(db.conn, run_id, Decimal("0.100001"))
    db.conn.rollback()


def test_negative_spend_is_rejected(db: Database) -> None:
    """The ledger is append-only and monotonic; a refund is not a budget concept."""
    run_id = db.insert_run()

    with pytest.raises(ValueError, match="negative"):
        record_spend(db.conn, run_id, Decimal("-1"))
    db.conn.rollback()
