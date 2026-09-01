"""The failure taxonomy and its disposition table.

Two kinds of test live here.

The first is ordinary: :func:`classify` maps each error onto its class, and anything
unrecognised onto ``PERMANENT``. The direction of that default matters — treating an
unknown bug as transient produces five slow, billed repetitions of the same crash.

The second is a *completeness* test, and it is the reason this file exists at all.
``DISPOSITIONS`` is the single source of truth for retry policy, consulted by both
``worker.py`` (run-level retries) and ``context.py`` (in-step retries). The realistic way
that breaks is not a wrong entry, it is a *missing* one: somebody adds a sixth
``ErrorClass``, forgets the table, and the first time that class is raised in production
the worker takes a ``KeyError`` while handling an error. So
:func:`test_every_error_class_has_a_disposition` is parametrised over ``ErrorClass``
itself — add a member without a disposition and CI fails before the code ships.

The backoff tests pin the shape of the wait rather than exact numbers: monotonic growth, a
hard ceiling, and jitter that actually varies. Jitter is not decoration — an unjittered
backoff synchronises every worker that got a 429 onto the same retry instant and
reproduces the overload that caused it.

A third kind arrived with the last review, in the section at the bottom: the **defensive
branches on the money path**. ``context._price``'s fallback to the requested model, the
replay guard that refuses to treat an unusable ledger row as a cache hit, and the cap on
what a repair prompt may quote back are all written to handle something that is not
supposed to happen. Nothing in the rest of the suite executes any of them, which makes them
the easiest code in the library to invert by accident: flip the fallback and every call from
a provider reporting a dated snapshot silently prices at zero, and a budget that stops
counting is worse than no budget at all. They live here because each one is about how llmrun
behaves when something has already gone wrong — the same question the taxonomy above
answers, one layer down. They need a real Postgres, like everything else in this suite.
"""

from __future__ import annotations

import random
from decimal import Decimal
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from llmrun import context
from llmrun.cache import input_hash
from llmrun.context import RunContext
from llmrun.cost import PRICES, cost_of
from llmrun.db import transaction
from llmrun.errors import (
    DISPOSITIONS,
    MAX_BACKOFF_SECONDS,
    BudgetExceededError,
    Disposition,
    ErrorClass,
    LLMRunError,
    MalformedOutputError,
    PermanentError,
    PolicyRefusalError,
    TransientError,
    backoff_delay,
    classify,
)
from llmrun.types import RUN_STATUSES, LLMResult

if TYPE_CHECKING:
    from conftest import Database

#: The disposition table exactly as the design contract states it. Written out by hand so
#: the assertion compares against the *document*, not against the implementation.
CONTRACT_TABLE = {
    ErrorClass.TRANSIENT: (True, 5, "failed"),
    ErrorClass.MALFORMED: (True, 2, "pending_review"),
    ErrorClass.POLICY: (False, 0, "pending_review"),
    ErrorClass.BUDGET: (False, 0, "halted_budget"),
    ErrorClass.PERMANENT: (False, 0, "failed"),
}


# --------------------------------------------------------------------------------------
# classify()
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (TransientError("upstream 503"), ErrorClass.TRANSIENT),
        (MalformedOutputError("not valid JSON"), ErrorClass.MALFORMED),
        (PolicyRefusalError("the model declined"), ErrorClass.POLICY),
        (BudgetExceededError("would exceed the cap"), ErrorClass.BUDGET),
        (PermanentError("bad argument"), ErrorClass.PERMANENT),
    ],
)
def test_classify_maps_each_llmrun_error_to_its_class(
    exception: LLMRunError, expected: ErrorClass
) -> None:
    assert classify(exception) == expected
    assert type(exception).error_class == expected


@pytest.mark.parametrize(
    "exception",
    [
        ValueError("a bug in pipeline code"),
        KeyError("missing_field"),
        TypeError("wrong argument type"),
        ZeroDivisionError("division by zero"),
        RuntimeError("something else entirely"),
    ],
)
def test_unknown_exceptions_are_permanent(exception: Exception) -> None:
    """Conservative on purpose: an unrecognised failure is assumed to be a bug.

    Assuming a bug is transient is how retry loops are born — five identical crashes, each
    one billed if the crash happens to be after the model call.
    """
    assert classify(exception) == ErrorClass.PERMANENT


def test_base_exceptions_are_permanent_too() -> None:
    """``classify`` takes ``BaseException``: a KeyboardInterrupt must not look retryable."""
    assert classify(KeyboardInterrupt()) == ErrorClass.PERMANENT


def test_every_error_is_an_llmrun_error() -> None:
    for error_type in (
        TransientError,
        MalformedOutputError,
        PolicyRefusalError,
        BudgetExceededError,
        PermanentError,
    ):
        assert issubclass(error_type, LLMRunError)


def test_error_class_values_round_trip_through_a_text_column() -> None:
    """``ErrorClass`` inherits from ``str`` so it can go straight into ``runs.error_class``.

    The column is ``text``; reading it back must compare equal to the enum member without
    anyone remembering to call ``.value``.
    """
    assert ErrorClass.TRANSIENT == "transient"
    assert ErrorClass("policy") is ErrorClass.POLICY
    assert str(ErrorClass.BUDGET) == "budget"


# --------------------------------------------------------------------------------------
# the disposition table
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("error_class", list(ErrorClass))
def test_every_error_class_has_a_disposition(error_class: ErrorClass) -> None:
    """Add an ``ErrorClass`` without a disposition and this test fails.

    That is the entire point of it. The worker looks the disposition up while it is already
    handling a failure, so a missing entry turns one classified error into an unhandled
    ``KeyError`` inside the error path — the least debuggable place it could possibly go.
    """
    assert error_class in DISPOSITIONS, (
        f"{error_class!r} has no entry in DISPOSITIONS. Every ErrorClass needs one: "
        "worker.py and context.py both look it up while handling a failure."
    )
    assert isinstance(DISPOSITIONS[error_class], Disposition)


def test_disposition_table_has_no_entries_for_unknown_classes() -> None:
    """The other direction: a stale entry means a class was renamed or removed."""
    assert set(DISPOSITIONS) == set(ErrorClass)


@pytest.mark.parametrize(("error_class", "expected"), sorted(CONTRACT_TABLE.items()))
def test_disposition_matches_the_design_contract(
    error_class: ErrorClass, expected: tuple[bool, int, str]
) -> None:
    retry, max_attempts, terminal_status = expected
    disposition = DISPOSITIONS[error_class]

    assert disposition.retry is retry
    assert disposition.max_attempts == max_attempts
    assert disposition.terminal_status == terminal_status


@pytest.mark.parametrize("error_class", list(ErrorClass))
def test_terminal_status_is_a_real_run_status(error_class: ErrorClass) -> None:
    """A terminal status that is not in ``RunStatus`` fails a CHECK in the reader, not here.

    Better to catch it in a unit test than to write an unparseable status into ``runs``.
    """
    assert DISPOSITIONS[error_class].terminal_status in RUN_STATUSES


@pytest.mark.parametrize("error_class", list(ErrorClass))
def test_non_retryable_classes_have_no_attempt_budget(error_class: ErrorClass) -> None:
    """``retry=False`` and ``max_attempts>0`` would be a contradiction the worker resolves
    silently, one way or the other. Keep the table unambiguous instead."""
    disposition = DISPOSITIONS[error_class]

    if not disposition.retry:
        assert disposition.max_attempts == 0
    else:
        assert disposition.max_attempts >= 1


def test_policy_and_budget_are_never_retried() -> None:
    """The two classes where a retry is not merely useless but actively expensive.

    Every retry of a refusal is billed and refused again; every retry past a budget cap is
    the exact event the cap exists to prevent.
    """
    assert DISPOSITIONS[ErrorClass.POLICY].retry is False
    assert DISPOSITIONS[ErrorClass.BUDGET].retry is False
    assert DISPOSITIONS[ErrorClass.BUDGET].terminal_status == "halted_budget"


# --------------------------------------------------------------------------------------
# backoff
# --------------------------------------------------------------------------------------


def test_backoff_growth_is_monotonic_with_a_seeded_rng() -> None:
    """Each wait is strictly longer than the one before it, jitter included.

    That is a property of *equal* jitter specifically: the delay for an attempt lands in
    ``[capped/2, capped)``, and because ``capped`` doubles, the bottom of the next window
    is the top of the previous one. Full jitter (``[0, capped)``) would not guarantee this,
    and a retry that waits less than its predecessor is not backing off.
    """
    transient = DISPOSITIONS[ErrorClass.TRANSIENT]
    rng = random.Random(20260812)

    delays = [backoff_delay(transient, attempt, rng=rng) for attempt in range(1, 6)]

    assert all(later > earlier for earlier, later in pairwise(delays)), (
        f"backoff must grow, got {delays}"
    )


def test_backoff_stays_inside_its_expected_window() -> None:
    transient = DISPOSITIONS[ErrorClass.TRANSIENT]
    base = transient.backoff_base_seconds
    rng = random.Random(7)

    for attempt in range(1, 6):
        uncapped = base * (2 ** (attempt - 1))
        delay = backoff_delay(transient, attempt, rng=rng)

        assert uncapped / 2 <= delay <= uncapped


def test_backoff_is_capped() -> None:
    """Without a ceiling a doubling sequence parks a worker for longer than its lease, and
    the run gets stolen mid-sleep by another worker."""
    transient = DISPOSITIONS[ErrorClass.TRANSIENT]
    rng = random.Random(11)

    assert backoff_delay(transient, 40, rng=rng) <= MAX_BACKOFF_SECONDS


def test_backoff_is_deterministic_for_a_given_seed() -> None:
    """Seeded runs reproduce exactly — which is what makes this testable at all."""
    transient = DISPOSITIONS[ErrorClass.TRANSIENT]

    first = [backoff_delay(transient, n, rng=random.Random(99)) for n in range(1, 4)]
    second = [backoff_delay(transient, n, rng=random.Random(99)) for n in range(1, 4)]

    assert first == second


def test_backoff_is_jittered() -> None:
    """Two workers with different random state must not retry at the same instant."""
    transient = DISPOSITIONS[ErrorClass.TRANSIENT]

    worker_a = backoff_delay(transient, 3, rng=random.Random(1))
    worker_b = backoff_delay(transient, 3, rng=random.Random(2))

    assert worker_a != worker_b


def test_non_retryable_dispositions_do_not_wait() -> None:
    rng = random.Random(3)

    for error_class in (ErrorClass.POLICY, ErrorClass.BUDGET, ErrorClass.PERMANENT):
        assert backoff_delay(DISPOSITIONS[error_class], 1, rng=rng) == 0.0


def test_malformed_repairs_immediately_instead_of_waiting() -> None:
    """A schema violation is not fixed by patience, so the repair attempt does not sleep.

    Waiting would only hold a worker slot open while the same prompt stayed just as wrong.
    """
    rng = random.Random(5)

    assert DISPOSITIONS[ErrorClass.MALFORMED].backoff_base_seconds == 0.0
    assert backoff_delay(DISPOSITIONS[ErrorClass.MALFORMED], 1, rng=rng) == 0.0


# --------------------------------------------------------------------------------------
# defensive branches on the money path
#
# Everything below drives a branch in `context.py` that only runs when something has
# already gone wrong. Each one is a handful of lines guarding a real failure, and each one
# can be inverted by a plausible-looking edit without a single other test noticing.
# --------------------------------------------------------------------------------------

#: A model with a real price, so a call through the offline fake accrues real spend. `echo`
#: is priced at zero, which makes it useless for anything asserting a cost is non-zero.
PRICED_MODEL = "claude-sonnet-5"

#: The dated snapshot a provider might answer under. Deliberately *not* in `PRICES` — the
#: whole point of `_price`'s fallback is a model id the table has never heard of.
UNPRICED_SNAPSHOT = "claude-sonnet-5-20260401"

STEP = "answer"
PROMPT = "Answer the payload as one JSON object."
PAYLOAD = {"subject": "quarterly costs"}
MAX_TOKENS = 64

#: Token counts big enough that the resulting cost cannot round to zero at `numeric(12,6)`:
#: at 3/15 USD per million, this is 0.010500 USD.
TOKENS_IN = 1000
TOKENS_OUT = 500

_SELECT_CALLS = """
    SELECT model, tokens_in, tokens_out, cost_usd, cached, response
    FROM llm_calls
    WHERE run_id = %s::uuid
    ORDER BY created_at
"""

#: Writes a billed ledger row by hand, so a test can put the ledger into a state no correct
#: writer produces. `_INSERT_LLM_CALL` in context.py is the only other writer of this table.
_INSERT_UNUSABLE_CALL = """
    INSERT INTO llm_calls (run_id, step_name, attempt, input_hash, model,
                           tokens_in, tokens_out, cost_usd, cached, response)
    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, false, %s::jsonb)
"""


class _RelabellingProvider:
    """Answers under a different model id than the one it was asked for.

    Not a contrived shape: this is what a provider does when it resolves an alias to the
    dated snapshot that actually served the request, and it is exactly the case `_price`'s
    fallback exists for.
    """

    name = "relabelling"
    model = PRICED_MODEL

    def __init__(self, reported_model: str) -> None:
        self.reported_model = reported_model
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
        self.calls += 1
        return LLMResult(
            text='{"ok":true}',
            model=self.reported_model,
            tokens_in=TOKENS_IN,
            tokens_out=TOKENS_OUT,
        )


def _context(db: Database, run_id: Any, provider: Any) -> RunContext:
    """A context driving a run directly, with no lease.

    ``worker_id=None`` is the documented "no lease exists" mode — right for a test that is
    exercising one branch rather than the worker loop, and the fencing checks are covered in
    ``test_worker.py`` where they belong.
    """
    return RunContext(
        run_id=run_id,
        tenant_id="errors",
        input={},
        conn=db.conn,
        provider=provider,
        default_model=PRICED_MODEL,
    )


def _round_zero_digest() -> str:
    """The ledger key `ctx.llm` will compute for the first round of the call below.

    Spelled out here rather than read back from the code so this is a statement of the
    contract in `RunContext.llm`: the key covers the effective parameters, which include
    ``max_tokens``, the prompt, and which repair round this is. The framework's own ``step``
    key is *not* in it — that is passed to the provider, not hashed — and a test that
    silently disagreed about any of this would build a row nothing ever looks up, then pass
    for the wrong reason.
    """
    return input_hash(
        STEP,
        PAYLOAD,
        PRICED_MODEL,
        {"max_tokens": MAX_TOKENS, "prompt": PROMPT, "repair_round": 0},
    )


def test_a_model_the_price_table_does_not_know_is_priced_as_the_one_requested(
    db: Database,
) -> None:
    """The ledger has to carry a real number when the provider relabels the model.

    ``_price`` looks the cost up by the id the provider *echoed back*, because that is what
    was actually billed. When that id has no price — a dated snapshot, a regional alias, a
    model registered under only one of its two names — it falls back to the id that was
    requested rather than giving up or, far worse, recording zero.

    Zero is the failure this pins. It reads as a successful, free call: ``spent_usd`` stays
    where it was, ``check_budget`` finds room forever, and the cap the whole library is built
    around never fires. Nothing raises, nothing logs an error, and the discrepancy surfaces
    as an invoice. So the assertion is not "the fallback ran" but "the recorded cost equals
    what the requested model costs, **and is not zero**" — the second half is the one that
    catches a fallback quietly rewritten to `Decimal(0)`.
    """
    assert UNPRICED_SNAPSHOT not in PRICES, (
        f"{UNPRICED_SNAPSHOT} has been given a price, so this test no longer exercises the "
        "fallback; pick another unregistered id"
    )

    run_id = db.insert_run(tenant_id="errors", status="running")
    provider = _RelabellingProvider(UNPRICED_SNAPSHOT)

    _context(db, run_id, provider).llm(STEP, PROMPT, PAYLOAD, max_tokens=MAX_TOKENS)

    with transaction(db.conn) as cur:
        cur.execute(_SELECT_CALLS, (str(run_id),))
        rows = cur.fetchall()

    assert provider.calls == 1
    assert len(rows) == 1
    row = rows[0]

    # The ledger records what actually served the request, not what was asked for: the
    # fallback changes how the call is *priced*, never what it claims happened.
    assert row["model"] == UNPRICED_SNAPSHOT
    assert row["cached"] is False

    expected = cost_of(PRICED_MODEL, TOKENS_IN, TOKENS_OUT)
    assert expected > Decimal("0"), "the fixture model must have a non-zero price"
    assert row["cost_usd"] == expected
    assert row["cost_usd"] > Decimal("0"), (
        "an unpriced reported model was billed at zero; the run now looks free, the budget "
        "cap can never fire, and the only place the truth appears is the provider's invoice"
    )
    # And the same number reached `runs.spent_usd`, which is what the cap is compared with.
    assert db.spent(run_id) == expected


def test_a_ledger_row_with_no_usable_text_is_not_treated_as_a_cache_hit(db: Database) -> None:
    """A replay is only safe when there is something to replay.

    ``_replay_with_cursor`` reads ``response->>'text'`` out of the matched row. If the
    document is not a mapping, or carries no string ``text``, the honest answer is "this is
    not a hit" — because the alternative is returning ``None``, or ``{}``, or an empty
    string to a pipeline that will checkpoint it as the step's output. A row like that is
    not produced by any current writer; it is produced by an older schema, a hand-edited
    row, or a partially written document, and every one of those is a reason to distrust it
    rather than to serve it. Calling the provider costs money; serving a wrong answer as a
    verified replay costs trust, and does it silently.

    The second half is about where the fresh call then *goes*. Migration 0003 made
    ``llm_calls_cache_idx`` UNIQUE on ``(run_id, input_hash) WHERE cached = false``, and the
    unusable row is already sitting on exactly the key the replacement needs. A plain INSERT
    is rejected there, which for one revision meant the provider was called, the money was
    spent, and the step then failed on a constraint violation — costing money *and* failing
    the run, which is strictly worse than the "costs money, not trust" trade the guard is
    written to make. ``_UPSERT_BILLED_CALL`` closes that: the unusable row is overwritten by
    the answer that replaces it, and ``cost_usd`` accumulates rather than being assigned,
    because the call that wrote the unusable row was billed too and a ledger that forgets
    one of two payments understates spend by exactly the amount the cap exists to catch.
    """
    run_id = db.insert_run(tenant_id="errors", status="running")

    with transaction(db.conn) as cur:
        cur.execute(
            _INSERT_UNUSABLE_CALL,
            (str(run_id), STEP, 0, _round_zero_digest(), PRICED_MODEL, 7, 3, 0, "{}"),
        )

    provider = _RelabellingProvider(PRICED_MODEL)
    answer = _context(db, run_id, provider).llm(STEP, PROMPT, PAYLOAD, max_tokens=MAX_TOKENS)

    # The guard did its job: the empty document was not served as an answer.
    assert provider.calls == 1, (
        "a ledger row with no replayable text was served as a cache hit; the pipeline just "
        "checkpointed a value that came from nowhere"
    )
    assert isinstance(answer, str) and answer, "the fresh call's answer was not returned"

    with transaction(db.conn) as cur:
        cur.execute(_SELECT_CALLS, (str(run_id),))
        rows = cur.fetchall()

    # One billed row, still — the replacement took the unusable row's place rather than
    # sitting beside it, which is what the UNIQUE index is there to guarantee.
    assert [row["cached"] for row in rows] == [False]
    assert rows[0]["response"] != {}, "the unusable document survived the replacement"
    assert rows[0]["response"]["text"] == answer

    # No `cached=true` row: the guard returns before recording a replay, so the ledger never
    # claims a saving that did not happen.
    assert db.count_llm_calls(run_id, cached=True) == 0

    # The step completed rather than failing on a constraint, and its checkpoint holds the
    # answer that was actually bought.
    steps = db.steps(run_id)
    assert [(step.name, step.status) for step in steps] == [(STEP, "succeeded")]


def test_the_repair_prompt_caps_both_of_its_untrusted_inputs() -> None:
    """A repair must be cheaper than the answer it is repairing, or it is not a repair.

    Two untrusted strings go into the follow-up prompt, and both are attacker-shaped in the
    same way — they are derived from model output. The rejected text was always capped. The
    validation error was not, and that omission quietly made the documented bound false:
    ``schema`` is arbitrary caller code, a pydantic ``ValidationError`` quotes back every
    value it rejected, so a 50KB runaway response produced a ~50KB error message and the
    repair prompt grew to hold it. The step then paid to send the runaway response *back* to
    the model, at input rates, to ask it to try again — the exact cost the repair loop exists
    to bound.

    The bound is asserted against ``_MAX_REPAIR_EXCERPT`` rather than a magic number, so
    retuning the cap does not need this test edited; ``slack`` covers the fixed scaffolding
    (the round header, the tags, the closing instruction) and nothing that scales with input.
    """
    original = "Summarise the thread as one JSON object."
    rejected = "x" * 50_000
    error = MalformedOutputError("y" * 50_000)

    prompt = context._repair_prompt(original, rejected, error, 1)

    slack = 500
    assert len(prompt) <= 2 * context._MAX_REPAIR_EXCERPT + len(original) + slack, (
        f"repair prompt is {len(prompt)} characters from {len(rejected)} of rejected text "
        f"and {len(str(error))} of validation error: an untrusted input is uncapped"
    )

    # Both really were cut, and both really are present — a cap that dropped one of them
    # entirely would satisfy the length bound while removing the model's reason to succeed.
    assert prompt.count("... [truncated]") == 2
    assert "<previous_reply>" in prompt and "<validation_error>" in prompt
    assert "x" * 100 in prompt
    assert "y" * 100 in prompt
    # The task itself is never truncated: the repair has to be answering the same question.
    assert original in prompt
