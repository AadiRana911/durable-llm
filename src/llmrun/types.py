"""Core value types shared by every other module.

These are deliberately dumb: frozen dataclasses with no database access and no I/O, so
they can be unit-tested without Postgres.

The one piece of real logic here is :meth:`RunRecord.from_row` / :meth:`StepRecord.from_row`.
Those are the *single* place where driver-shaped data (``jsonb``, ``numeric``, ``uuid``,
``timestamptz``) is turned into Python-shaped data. Keeping the conversion in one place is
what lets the rest of the codebase assume, without defensive checks at every call site,
that money is always :class:`decimal.Decimal` and never ``float`` — a float dollar amount
that survives two additions and a round-trip through JSON is a support ticket, not a
rounding error.

``psycopg`` 3 already returns ``Decimal`` for ``numeric``, ``UUID`` for ``uuid``, and
decoded Python objects for ``jsonb``. The coercions below are therefore normally
no-ops; they exist so that a row coming from a differently-configured cursor, a JSON
fixture, or a hand-built test dict still produces an identical record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Final, Literal, get_args
from uuid import UUID

__all__ = [
    "RUN_STATUSES",
    "STEP_STATUSES",
    "LLMResult",
    "Row",
    "RunRecord",
    "RunStatus",
    "StepRecord",
    "StepStatus",
    "as_decimal",
    "as_decimal_or_zero",
]

RunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "pending_review",
    "halted_budget",
]
StepStatus = Literal["running", "succeeded", "failed"]

#: Every legal value of ``runs.status``. Derived from the ``Literal`` so the type and the
#: runtime tuple can never drift apart; the CLI uses it for ``--status`` choices.
RUN_STATUSES: Final[tuple[RunStatus, ...]] = get_args(RunStatus)
#: Every legal value of ``steps.status``.
STEP_STATUSES: Final[tuple[StepStatus, ...]] = get_args(StepStatus)

#: A single database row as produced by ``psycopg.rows.dict_row``.
Row = Mapping[str, Any]


# --------------------------------------------------------------------------------------
# row coercion helpers
# --------------------------------------------------------------------------------------


def _require(row: Row, field: str) -> Any:
    """Fetch a column, failing with a message that says which record wanted it."""
    try:
        return row[field]
    except KeyError:
        raise KeyError(f"row is missing required column {field!r}") from None


def _as_uuid(value: Any, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return UUID(bytes=bytes(value))
    raise TypeError(f"{field}: expected a uuid, got {type(value).__name__}")


def _as_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"{field}: expected a timestamptz, got {type(value).__name__}")


def _as_optional_datetime(value: Any, field: str) -> datetime | None:
    return None if value is None else _as_datetime(value, field)


def as_decimal(value: Any) -> Decimal | None:
    """Coerce a money value to :class:`~decimal.Decimal`, preserving ``None``.

    The lenient counterpart to :func:`_as_decimal`: that one is for *columns*, where a
    surprising type is a schema problem worth raising over, while this one is for values
    arriving from anywhere — a caller's argument, a hand-built fixture, a driver that was
    configured differently — where the useful behaviour is to coerce and carry on.

    Always through ``str()``, never ``Decimal(float)``. ``Decimal(0.1)`` is
    ``0.1000000000000000055511151231257827``; ``Decimal("0.1")`` is what the database
    printed. Routing every money value through one function is what stops that distinction
    being re-derived, slightly differently, in each module that needs it — which is what had
    happened: ``budget._to_decimal`` and ``context._as_decimal`` were the same nine lines
    with different names, and ``types.py``'s docstring claimed to be the single conversion
    point while both of them bypassed it.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def as_decimal_or_zero(value: Any) -> Decimal:
    """Like :func:`as_decimal`, treating a missing value as zero rather than ``None``.

    For the columns the schema declares ``NOT NULL DEFAULT 0`` — ``spent_usd``, ``cost_usd``
    — where ``None`` can only come from a fixture that predates the column, and where
    treating it as zero is both correct and what every call site wanted anyway.
    """
    coerced = as_decimal(value)
    return Decimal(0) if coerced is None else coerced


def _as_decimal(value: Any, field: str) -> Decimal:
    """Coerce a money column to ``Decimal`` — never via ``Decimal(float)``."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str | float):
        # str() first: Decimal(0.1) is 0.1000000000000000055511151231257827, Decimal("0.1")
        # is exactly what the database printed.
        return Decimal(str(value))
    raise TypeError(f"{field}: expected a numeric, got {type(value).__name__}")


def _as_optional_decimal(value: Any, field: str) -> Decimal | None:
    return None if value is None else _as_decimal(value, field)


def _as_json(value: Any) -> Any:
    """Normalise a ``jsonb`` column to a plain Python object.

    Only raw driver bytes are decoded. A ``str`` is left alone on purpose: a ``jsonb``
    column holding the JSON document ``"hello"`` decodes to the Python string ``hello``,
    and re-parsing it would silently turn the document ``"123"`` into the integer ``123``.
    """
    if isinstance(value, bytes | bytearray | memoryview):
        return json.loads(bytes(value).decode("utf-8"))
    return value


def _as_json_object(value: Any, field: str) -> dict[str, Any]:
    """Normalise a ``jsonb`` column that is contractually a JSON object.

    Here a ``str`` *is* parsed: the column can only ever hold an object, so a bare string
    is an encoding artefact rather than a legitimate value.
    """
    if value is None:
        return {}
    if isinstance(value, bytes | bytearray | memoryview):
        value = json.loads(bytes(value).decode("utf-8"))
    elif isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field}: expected a json object, got {type(value).__name__}")
    return dict(value)


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field}: expected an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{field}: expected an integer, got {type(value).__name__}")


def _as_str(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"{field}: expected text, got {type(value).__name__}")


def _as_optional_str(value: Any, field: str) -> str | None:
    return None if value is None else _as_str(value, field)


def _as_run_status(value: Any) -> RunStatus:
    status = _as_str(value, "status")
    if status not in RUN_STATUSES:
        raise ValueError(f"unknown run status {status!r}; expected one of {RUN_STATUSES}")
    return status  # type: ignore[return-value]


def _as_step_status(value: Any) -> StepStatus:
    status = _as_str(value, "status")
    if status not in STEP_STATUSES:
        raise ValueError(f"unknown step status {status!r}; expected one of {STEP_STATUSES}")
    return status  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    """One provider completion.

    Frozen because it is written to the ledger verbatim: if the object that produced an
    ``llm_calls`` row could still be mutated afterwards, the cost ledger and the thing it
    claims to describe could disagree.
    """

    text: str
    model: str
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class RunRecord:
    """A row of ``runs``: one execution of one pipeline for one tenant."""

    id: UUID
    pipeline: str
    tenant_id: str
    status: RunStatus
    input: dict
    output: Any | None
    budget_usd: Decimal | None
    spent_usd: Decimal
    error_class: str | None
    error_detail: str | None
    #: Claims. Incremented by the worker's claim query, including when a lease expired and
    #: the run was stolen back, so it counts *executions started* rather than failures.
    attempt: int
    #: Recorded failures. Incremented only when a worker writes a verdict, which is what
    #: makes it the honest thing to measure a retry budget against — see migrations/0002.
    failures: int
    #: The execution ceiling this run was enqueued under, copied from the pipeline at
    #: enqueue time. Carried on the record because the claim query enforces it in SQL
    #: (``WHERE attempt < max_attempts``) and the requeue decision has to enforce the *same*
    #: number: a worker that read its ceiling off the live ``@pipeline`` decorator instead
    #: would disagree with the claim query the moment a deploy changed it, and requeue a run
    #: into a state the claim query will not take back.
    max_attempts: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Row) -> RunRecord:
        """Build a record from a ``dict_row`` mapping.

        Extra columns (``locked_by``, ``locked_at`` from the worker's ``RETURNING *``)
        are ignored, so the same constructor serves every query that selects at least the
        columns below.
        """
        spent = _as_optional_decimal(_require(row, "spent_usd"), "spent_usd")
        return cls(
            id=_as_uuid(_require(row, "id"), "id"),
            pipeline=_as_str(_require(row, "pipeline"), "pipeline"),
            tenant_id=_as_str(_require(row, "tenant_id"), "tenant_id"),
            status=_as_run_status(_require(row, "status")),
            input=_as_json_object(_require(row, "input"), "input"),
            output=_as_json(row.get("output")),
            budget_usd=_as_optional_decimal(row.get("budget_usd"), "budget_usd"),
            # NOT NULL DEFAULT 0 in the schema; the fallback only fires for hand-built rows.
            spent_usd=Decimal(0) if spent is None else spent,
            error_class=_as_optional_str(row.get("error_class"), "error_class"),
            error_detail=_as_optional_str(row.get("error_detail"), "error_detail"),
            attempt=_as_int(_require(row, "attempt"), "attempt"),
            # NOT NULL DEFAULT 0 in the schema, so `.get` with a fallback rather than
            # `_require`: a hand-built row in a test that predates the column still builds.
            failures=_as_int(row.get("failures", 0), "failures"),
            # Same treatment as `failures`, and for the same reason: the column arrived in
            # 0002, so a hand-built row in a test may not carry it. The fallback matches the
            # column default and `pipeline.DEFAULT_MAX_ATTEMPTS`; all three have to agree.
            max_attempts=_as_int(row.get("max_attempts", 5), "max_attempts"),
            created_at=_as_datetime(_require(row, "created_at"), "created_at"),
            updated_at=_as_datetime(_require(row, "updated_at"), "updated_at"),
        )


@dataclass(frozen=True)
class StepRecord:
    """A row of ``steps``: one checkpointed unit of work inside a run.

    ``input_hash`` is the resumption key. A stored step is replayed instead of re-executed
    only when its hash still matches the hash of the arguments it is being called with —
    see the ``step()`` semantics in the design contract.
    """

    run_id: UUID
    name: str
    status: StepStatus
    input_hash: str
    output: Any | None
    attempt: int
    error_class: str | None
    error_detail: str | None
    started_at: datetime
    ended_at: datetime | None

    @classmethod
    def from_row(cls, row: Row) -> StepRecord:
        """Build a record from a ``dict_row`` mapping."""
        return cls(
            run_id=_as_uuid(_require(row, "run_id"), "run_id"),
            name=_as_str(_require(row, "name"), "name"),
            status=_as_step_status(_require(row, "status")),
            input_hash=_as_str(_require(row, "input_hash"), "input_hash"),
            output=_as_json(row.get("output")),
            attempt=_as_int(_require(row, "attempt"), "attempt"),
            error_class=_as_optional_str(row.get("error_class"), "error_class"),
            error_detail=_as_optional_str(row.get("error_detail"), "error_detail"),
            started_at=_as_datetime(_require(row, "started_at"), "started_at"),
            ended_at=_as_optional_datetime(row.get("ended_at"), "ended_at"),
        )
