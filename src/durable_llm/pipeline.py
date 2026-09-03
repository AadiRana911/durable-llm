"""The ``@pipeline`` decorator and the :class:`Pipeline` object it produces.

A pipeline is an ordinary Python function that takes a :class:`~durable_llm.context.RunContext`
as its first argument. There is no DSL, no graph, no YAML: control flow is ``if`` and
``for``, because the moment durable execution requires its own dialect, the cost of using
it exceeds the cost of the crash it was protecting you from.

The decorator adds exactly three things to the function:

* a **name**, so a row in ``runs`` can point at it,
* a **budget**, so a run has a cap it cannot talk its way past,
* an **attempt limit**, consulted by the worker when a failure is retryable.

Everything else — checkpointing, caching, spend accounting — happens through the context
the function is handed, not through the decorator. That keeps the decorated function
directly callable in a unit test with a stub context, which is the property that makes
pipelines testable without a database.

Enqueueing is deliberately a *write and nothing else*. It inserts one row with status
``queued`` and returns; no work starts, no connection is held, no in-process queue exists.
The worker is what turns rows into execution, so an enqueue that has committed will
eventually run even if the process that made it dies a millisecond later.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Final, overload
from uuid import UUID

from durable_llm.db import connect, transaction
from durable_llm.registry import register

if TYPE_CHECKING:  # pragma: no cover - typing only
    import psycopg
    from psycopg.rows import DictRow

    from durable_llm.context import RunContext

__all__ = ["DEFAULT_MAX_ATTEMPTS", "Pipeline", "pipeline"]

logger = logging.getLogger("durable_llm.pipeline")

#: Run-level attempts when a pipeline does not say otherwise. It bounds two things at once:
#: how many times a worker may *execute* the run (checked by the claim query, so a run that
#: kills its worker before it can record anything is still bounded), and how many recorded
#: failures it may accumulate before a retryable failure becomes terminal. The in-step retry
#: counts live in ``DISPOSITIONS`` and are a separate thing entirely.
#:
#: Kept in sync with the ``runs.max_attempts`` column default in migrations/0002.
DEFAULT_MAX_ATTEMPTS: Final[int] = 5

_QUEUED: Final[str] = "queued"

# `input` is passed as text with an explicit ::jsonb cast rather than as a psycopg Json
# wrapper, so the exact bytes stored are the exact bytes this module serialised. Every
# value is a parameter; there is no string interpolation anywhere near this statement.
#
# `max_attempts` is copied onto the row rather than looked up at claim time because the
# claim is a single SQL statement and SQL cannot ask a Python object how many attempts a
# pipeline allows. Copying it also pins the ceiling a run was enqueued under, so raising
# the limit tomorrow does not retroactively grant more attempts to a run failing today.
_INSERT_RUN: Final[str] = """
    INSERT INTO runs (pipeline, tenant_id, status, input, budget_usd, max_attempts)
    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
    RETURNING id
"""


class Pipeline:
    """A named, budgeted, enqueueable pipeline function.

    Instances carry the wrapped function's ``__name__``, ``__doc__``, ``__module__`` and
    ``__wrapped__`` (via :func:`functools.update_wrapper`), so ``help()``,
    ``inspect.signature()`` and Sphinx all still see the function the author wrote rather
    than this wrapper.

    Attributes:
        name: Identifier stored in ``runs.pipeline`` and passed to ``durable-llm enqueue``.
        fn: The undecorated function.
        budget_usd: Hard spend cap copied onto every run this pipeline enqueues. ``None``
            means unlimited, which is a deliberate opt-in rather than the default of
            convenience.
        max_attempts: The run-level ceiling, copied onto every run this pipeline enqueues.
            It caps executions (enforced in the claim query, because a killed worker never
            reaches the code that records a verdict) and, separately, recorded failures — a
            retryable failure is requeued only while the run has failures left. Splitting
            the two is what keeps a lease expiry from spending a retry the run never used.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        budget_usd: str | int | float | Decimal | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"@pipeline expects a callable, got {type(fn).__name__}")

        # update_wrapper copies fn.__dict__ over ours, so it has to run *before* we set our
        # own attributes; otherwise a function carrying stray attributes could overwrite
        # them. Doing it at all is what preserves the functools.wraps contract.
        functools.update_wrapper(self, fn)

        self.fn = fn
        self.name = name or getattr(fn, "__name__", "")
        if not self.name:
            raise ValueError("cannot name this pipeline; pass name= to @pipeline")

        self.budget_usd = coerce_budget_usd(budget_usd)
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    # -- execution -----------------------------------------------------------------

    def __call__(self, ctx: RunContext, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly.

        Decorating a function must not take it away from its author: ``summarise_thread(ctx,
        thread_id="t1")`` keeps working, which is what lets a pipeline be unit-tested with a
        fake context and no database at all.
        """
        return self.fn(ctx, *args, **kwargs)

    def execute(self, ctx: RunContext) -> Any:
        """Run the pipeline with the arguments stored on the run. Used by the worker.

        ``ctx.input`` is the ``kwargs`` that :meth:`enqueue` serialised, round-tripped
        through ``jsonb``. A mismatch between those keys and the function's parameters
        raises ``TypeError``, which :func:`durable_llm.errors.classify` treats as ``PERMANENT``:
        a run enqueued with the wrong arguments will never succeed, so retrying it five
        times would only cost five times as much to learn the same thing.
        """
        return self.fn(ctx, **ctx.input)

    # -- enqueueing ----------------------------------------------------------------

    def enqueue(
        self,
        *,
        tenant_id: str,
        conn: psycopg.Connection[DictRow] | None = None,
        **kwargs: Any,
    ) -> UUID:
        """Insert a ``queued`` run and return its id.

        Every keyword other than ``tenant_id`` and ``conn`` becomes the run's input, stored
        as ``jsonb`` so the arguments survive the process that supplied them.

        Args:
            tenant_id: Owner of the run. Required, and required to be non-empty: it is the
                only thing separating one customer's spend from another's in the cost
                report, and a run that lands in the wrong tenant is not something the
                report can fix later.
            conn: An existing connection. When omitted, one is opened from
                ``DURABLE_LLM_DATABASE_URL`` and closed again before returning.
            **kwargs: The pipeline function's arguments. Must be JSON-serialisable.

        Returns:
            The new run's id.

        Note:
            The insert is **committed** before this returns, including when the caller
            supplied the connection. An uncommitted enqueue is invisible to every worker,
            so returning an id for a row no one else can see would be a promise this
            library cannot keep. Callers who need the enqueue to be atomic with their own
            writes should do those writes on the same connection *before* calling.
        """
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required and must be a non-empty string")

        # Serialise before opening anything: a non-serialisable argument is a caller bug,
        # and it should surface as a TypeError here rather than as a failed run later.
        payload = _encode_input(self.name, kwargs)

        connection = conn if conn is not None else connect()
        row: Any = None
        try:
            with transaction(connection) as cur:
                cur.execute(
                    _INSERT_RUN,
                    (
                        self.name,
                        tenant_id,
                        _QUEUED,
                        payload,
                        self.budget_usd,
                        self.max_attempts,
                    ),
                )
                row = cur.fetchone()
            # Committed here. A crash immediately before this line loses the enqueue
            # entirely — nothing was promised, because this method had not returned an id.
            # A crash immediately after loses nothing at all: the row is durable and the
            # next worker to poll will claim it. Those are the only two outcomes, which is
            # precisely why enqueue does no work beyond the insert.
        finally:
            if conn is None:
                connection.close()

        if row is None:  # pragma: no cover - RETURNING always yields a row on success
            raise RuntimeError(f"INSERT into runs returned no id for pipeline {self.name!r}")

        run_id = _row_uuid(row)
        logger.info(
            "enqueued run %s: pipeline=%s tenant=%s budget=%s",
            run_id,
            self.name,
            tenant_id,
            "unlimited" if self.budget_usd is None else f"{self.budget_usd} USD",
        )
        return run_id

    def __repr__(self) -> str:
        budget = "unlimited" if self.budget_usd is None else str(self.budget_usd)
        return f"<Pipeline {self.name!r} budget_usd={budget} max_attempts={self.max_attempts}>"


# Two overloads, because the union return type below is honest about the runtime and useless
# to a type checker: given `Pipeline | Callable[..., Pipeline]` it cannot tell which arm a
# particular call site got, so BOTH documented spellings fail — bare `@pipeline` leaves the
# decorated name typed as a union, and `@pipeline(name=...)` is reported as "object is not
# callable" when the decorator is applied. This package ships py.typed and advertises
# `Typing :: Typed`, so that lands in users' checkers, not ours.
#
# The discriminator is `fn`: supplied means the bare form and the result is the Pipeline
# itself; omitted means the call form and the result is the decorator that will produce one.
# The union survives only on the implementation signature, which no caller ever sees.
@overload
def pipeline(fn: Callable[..., Any]) -> Pipeline: ...


@overload
def pipeline(
    fn: None = None,
    *,
    name: str | None = None,
    budget_usd: str | int | float | Decimal | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Callable[[Callable[..., Any]], Pipeline]: ...


def pipeline(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    budget_usd: str | int | float | Decimal | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Pipeline | Callable[[Callable[..., Any]], Pipeline]:
    """Turn a function into a registered :class:`Pipeline`.

    Usable bare or with arguments::

        @pipeline
        def summarise_thread(ctx, thread_id): ...


        @pipeline(name="summarise_thread", budget_usd="2.50", max_attempts=5)
        def summarise(ctx, thread_id): ...

    Registration happens at import time, which is what makes ``--import`` the entire
    deployment story: a worker knows how to run whatever its imports defined.

    Args:
        fn: The function, when used bare.
        name: Registered name. Defaults to the function's ``__name__``.
        budget_usd: Spend cap per run. Accepts ``str``/``int``/``float``/``Decimal``/``None``.
        max_attempts: Run-level attempt limit for retryable failures.

    Raises:
        ValueError: Another pipeline already claims ``name``.
    """

    def decorate(target: Callable[..., Any]) -> Pipeline:
        built = Pipeline(target, name=name, budget_usd=budget_usd, max_attempts=max_attempts)
        register(built.name, built)
        return built

    return decorate(fn) if fn is not None else decorate


def coerce_budget_usd(value: str | int | float | Decimal | None) -> Decimal | None:
    """Coerce a budget to :class:`~decimal.Decimal`, never via ``Decimal(float)``.

    ``Decimal(2.50)`` is ``2.5000000000000000e+0``-ish — 2.5 exactly, as it happens, but
    ``Decimal(0.1)`` is ``0.1000000000000000055511151231257827``. Routing everything through
    ``str()`` first means the cap stored in Postgres is the cap the author typed, which is
    the only version of the number anyone will defend in a meeting.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        budget = value
    else:
        try:
            budget = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"budget_usd must be a number, got {value!r}") from exc

    if not budget.is_finite():
        raise ValueError(f"budget_usd must be a finite amount, got {value!r}")
    if budget < 0:
        raise ValueError(f"budget_usd must not be negative, got {value!r}")
    return budget


def _encode_input(name: str, kwargs: dict[str, Any]) -> str:
    """Serialise a run's arguments to JSON text for the ``input`` column.

    No ``default=`` fallback here, on purpose — not even the type-tagged one
    :func:`durable_llm.cache.canonical_json` uses. A fallback is defensible there because the
    result is only ever a *key*: the tag stops two different values collapsing onto one
    key, and the worst that remains is a stale key that misses the cache and re-bills.
    Here the JSON *is* the argument list the worker will later call the function with, so
    reducing an object to text would hand the pipeline a string where it expected the
    object and produce a wrong answer instead of an error. No tag repairs that — the value
    has to come back out of ``jsonb`` as itself, and only plain JSON types do.

    ``allow_nan=False`` for a related reason. Python's encoder emits the bare tokens ``NaN``,
    ``Infinity`` and ``-Infinity`` by default, none of which are JSON and none of which
    ``jsonb`` will parse — so a ``float("nan")`` argument sails past this guard and dies
    several layers down as a raw ``psycopg.errors.InvalidTextRepresentation`` on the INSERT,
    with a message about SQL rather than about the argument the caller passed. Rejecting it
    here is what makes the promise above ("a TypeError here rather than a failed run later")
    true for the one non-serialisable value ``json.dumps`` does not otherwise complain about.
    """
    try:
        return json.dumps(kwargs, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"arguments for pipeline {name!r} must be JSON-serialisable so they can be "
            f"stored in runs.input, and finite (NaN and Infinity have no JSON form that "
            f"jsonb accepts): {exc}"
        ) from exc


def _row_uuid(row: Any) -> UUID:
    """Read the id out of a ``RETURNING id`` row, whatever row factory produced it."""
    value = row["id"] if isinstance(row, dict) else row[0]
    return value if isinstance(value, UUID) else UUID(str(value))
