"""The failure taxonomy — the centrepiece of durable_llm.

Most production LLM code contains some variant of::

    try:
        result = call_the_model(prompt)
    except Exception:
        retry()

That single ``except`` is the most expensive line in the file, because "the call failed"
is not one situation. It is at least five, and they want opposite responses:

``TRANSIENT``
    A timeout, a 429, a 5xx, a reset connection. Nothing is wrong with the request. The
    only correct response is to try again later, spaced out with exponential backoff and
    jitter so that a provider blip does not turn into a self-inflicted thundering herd.
    Retrying is *free* in the sense that a failed call was never billed.

``MALFORMED``
    The model answered, we were billed, and the answer did not satisfy the schema. A plain
    retry sends the identical prompt and tends to get an identically-shaped failure — you
    pay twice for the same mistake. What actually works is a *repair* attempt: feed the
    validation error back into the prompt so the model can correct itself. Two attempts,
    then stop, because a model that has failed the same schema twice is usually being
    asked for something the schema does not allow.

``POLICY``
    The provider refused, or the pipeline escalated deliberately. Retrying a refusal is
    the purest form of burning money: every attempt is billed, every attempt is refused,
    and the loop only stops when the budget does. Refusals are a routing decision, not an
    error to paper over — the run ends in ``pending_review`` so a human sees it.

``BUDGET``
    The projected spend exceeds the cap. Retrying is not merely useless, it is the exact
    behaviour the cap exists to prevent. The run halts in its own terminal status
    (``halted_budget``) so that "we stopped you" is never confused with "it broke".

``PERMANENT``
    A bug in the pipeline code — a ``KeyError``, a bad argument, a typo in a model name.
    No amount of retrying fixes a program. Fail fast and loudly so it gets fixed.

Collapsing these into one handler produces the two classic failure modes: a refusal loop
that quietly spends the monthly budget in an afternoon, and a malformed-output loop that
retries an unrepaired prompt until it gives up. Keeping them distinct costs one enum.

The policy itself lives in :data:`DISPOSITIONS`, deliberately as *data* rather than as
``if`` statements. ``worker.py`` (run-level retries) and ``context.py`` (in-step retries)
both consult the same table, so the retry policy cannot drift between them, and the table
can be asserted against the design contract in a three-line test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Final

from durable_llm.types import RunStatus

__all__ = [
    "DISPOSITIONS",
    "IN_STEP_RETRY_CLASSES",
    "MAX_BACKOFF_SECONDS",
    "MAX_ERROR_DETAIL",
    "BudgetExceededError",
    "Disposition",
    "ErrorClass",
    "LLMRunError",
    "MalformedOutputError",
    "PermanentError",
    "PolicyRefusalError",
    "TransientError",
    "backoff_delay",
    "classify",
    "error_detail",
]


class ErrorClass(str, Enum):
    """How a failure should be treated. Stored verbatim in ``runs.error_class``.

    Inherits from ``str`` so the value can go straight into a parameterised SQL statement
    and come back out of a ``text`` column comparing equal.
    """

    TRANSIENT = "transient"
    MALFORMED = "malformed"
    POLICY = "policy"
    BUDGET = "budget"
    PERMANENT = "permanent"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class LLMRunError(Exception):
    """Base of the taxonomy.

    The default is :attr:`ErrorClass.PERMANENT`, matching :func:`classify`'s treatment of
    unknown exceptions: anything whose failure mode has not been explicitly reasoned about
    is assumed to be a bug, because *assuming a bug is transient is how retry loops are
    born*.
    """

    error_class: ClassVar[ErrorClass] = ErrorClass.PERMANENT


class TransientError(LLMRunError):
    """Timeout, 429, 5xx, connection reset — worth trying again, unchanged."""

    error_class: ClassVar[ErrorClass] = ErrorClass.TRANSIENT


class MalformedOutputError(LLMRunError):
    """The response did not satisfy ``schema``; needs a repair prompt, not a plain retry."""

    error_class: ClassVar[ErrorClass] = ErrorClass.MALFORMED


class PolicyRefusalError(LLMRunError):
    """The provider refused, or the pipeline escalated. Retrying is pointless and billed."""

    error_class: ClassVar[ErrorClass] = ErrorClass.POLICY


class BudgetExceededError(LLMRunError):
    """Projected spend exceeds the cap. Raised *before* the call, never after."""

    error_class: ClassVar[ErrorClass] = ErrorClass.BUDGET


class PermanentError(LLMRunError):
    """A bug in pipeline code. Retrying a program does not debug it."""

    error_class: ClassVar[ErrorClass] = ErrorClass.PERMANENT


#: Cap on the ``error_detail`` columns. Both are unbounded ``text``, but a multi-megabyte
#: provider payload pasted into an error message turns ``durable-llm runs list`` into a memory
#: problem, and nothing after the first few lines has ever helped anyone.
MAX_ERROR_DETAIL: Final[int] = 4000


def error_detail(exc: BaseException, limit: int = MAX_ERROR_DETAIL) -> str:
    """Bounded, human-first rendering of a failure for an ``error_detail`` column.

    The exception *type* is kept alongside the message: ``KeyError: 'thread_id'`` says what
    went wrong, where a bare ``'thread_id'`` says nothing at all.

    One implementation rather than two. ``worker.py`` and ``context.py`` each grew their own,
    with the same cap and different truncation — one appended ``"... [truncated]"`` past the
    limit and the other replaced the last three characters with ``"..."``, so the same
    exception was recorded at two different lengths depending on which module caught it, and
    a test pinning one of them said nothing about the other.
    """
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) <= limit:
        return text
    marker = "... [truncated]"
    return text[: max(0, limit - len(marker))] + marker


def classify(exc: BaseException) -> ErrorClass:
    """Map any exception onto the taxonomy.

    Known durable_llm errors report their own class; everything else is ``PERMANENT``. The
    conservative direction matters: mislabelling a real bug as transient produces five
    slow, billed repetitions of the same crash, whereas mislabelling a transient blip as
    permanent produces one honest failure that a human can see and requeue.
    """
    if isinstance(exc, LLMRunError):
        return exc.error_class
    return ErrorClass.PERMANENT


@dataclass(frozen=True)
class Disposition:
    """What to do about one class of failure.

    ``max_attempts`` counts *total* attempts, not extra ones: ``5`` means the work is
    tried at most five times in all. Non-retryable classes carry ``0`` to make
    "there is no attempt budget here" unambiguous.
    """

    retry: bool
    max_attempts: int
    terminal_status: RunStatus
    backoff_base_seconds: float = 1.0


#: The disposition table from the design contract, in one place, as data.
#:
#: ``malformed`` carries a zero backoff on purpose: its retry strategy is a repair prompt,
#: not patience. Waiting changes nothing about a schema violation, and the run is holding
#: a worker slot while it waits.
DISPOSITIONS: Final[dict[ErrorClass, Disposition]] = {
    ErrorClass.TRANSIENT: Disposition(
        retry=True,
        max_attempts=5,
        terminal_status="failed",
        backoff_base_seconds=1.0,
    ),
    ErrorClass.MALFORMED: Disposition(
        retry=True,
        max_attempts=2,
        terminal_status="pending_review",
        backoff_base_seconds=0.0,
    ),
    ErrorClass.POLICY: Disposition(
        retry=False,
        max_attempts=0,
        terminal_status="pending_review",
    ),
    ErrorClass.BUDGET: Disposition(
        retry=False,
        max_attempts=0,
        terminal_status="halted_budget",
    ),
    ErrorClass.PERMANENT: Disposition(
        retry=False,
        max_attempts=0,
        terminal_status="failed",
    ),
}

#: Classes whose ``max_attempts`` budget is spent *inside the step*, not by the worker.
#:
#: The disposition table in the design contract carries a "strategy" column that the
#: :class:`Disposition` dataclass has no field for, and for ``malformed`` that column reads
#: "repair prompt **(in-step)**". This constant is where that column lives, because without
#: it ``max_attempts`` means two different things depending on which loop reads it: a
#: ``MalformedOutputError`` only ever escapes ``ctx.llm()`` after the repair loop has
#: already used its two attempts, so a worker that then requeued the whole run would buy two
#: more repairs — four billed attempts for a table that says two — and delay
#: ``pending_review`` by a whole extra pass over the pipeline.
#:
#: ``transient`` is deliberately absent: it is also retried in-step, but it can be raised by
#: ordinary ``ctx.step()`` code that never goes near the in-step retry loop, so the
#: run-level budget is the only one such a failure ever gets.
IN_STEP_RETRY_CLASSES: Final[frozenset[ErrorClass]] = frozenset({ErrorClass.MALFORMED})

#: Ceiling for a single backoff wait. Without it, attempt 5 of a doubling sequence with a
#: larger base can park a worker for longer than the lease, and the run gets stolen
#: mid-sleep by another worker.
MAX_BACKOFF_SECONDS: Final[float] = 60.0


def backoff_delay(
    disposition: Disposition,
    attempt: int,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before the attempt following ``attempt``.

    Exponential backoff with *equal jitter*: the wait doubles per attempt, is clamped to
    :data:`MAX_BACKOFF_SECONDS`, and then half of it is randomised::

        delay = capped * (0.5 + 0.5 * rng.random())

    Jitter is not decoration. When a provider returns 429 to a hundred concurrent workers,
    an unjittered backoff synchronises all hundred onto the same retry instant and
    reproduces the overload that caused the 429. Equal jitter spreads them across a window
    while still guaranteeing a non-zero wait, which full jitter does not.

    ``attempt`` is the 1-based number of the attempt that just failed; ``0`` and ``1`` both
    yield the base delay, so callers counting from either end behave sanely. Pass ``rng``
    (a seeded :class:`random.Random`) to make the result deterministic in tests.

    Non-retryable dispositions, and any disposition with a non-positive base, return
    ``0.0`` — there is nothing to wait for.
    """
    if not disposition.retry or disposition.backoff_base_seconds <= 0:
        return 0.0

    exponent = max(attempt - 1, 0)
    raw = disposition.backoff_base_seconds * (2**exponent)
    capped = min(raw, MAX_BACKOFF_SECONDS)
    jitter = rng.random() if rng is not None else random.random()
    return capped * (0.5 + 0.5 * jitter)
