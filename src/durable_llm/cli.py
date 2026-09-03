"""``durable-llm`` — the command line interface.

Seven commands, one job each::

    durable-llm migrate                       apply migrations/*.sql
    durable-llm worker                        claim queued runs and execute them
    durable-llm enqueue PIPELINE --tenant T   put a run on the queue
    durable-llm runs list                     what has run, and how it ended
    durable-llm runs show RUN_ID              one run in full: steps and the cost ledger
    durable-llm costs                         spend by pipeline and model, and what caching saved
    durable-llm health                        is the database reachable, and what is backed up

``runs show`` is the command this project exists to be able to print. It puts the step
timeline next to the per-call cost ledger and marks every call that was replayed from the
ledger instead of re-purchased, so "killing the worker mid-run cost nothing the second time"
is something a reviewer reads off the screen rather than takes on trust.

Conventions in here:

* Exit codes: ``0`` success, ``1`` the command failed, ``2`` bad usage.
* Results go to stdout via :func:`_emit`; progress and diagnostics go to stderr via
  :mod:`logging`. That split is what lets ``run_id=$(durable-llm enqueue ...)`` work while the
  operator still sees what happened.
* Tables are laid out with ``str.ljust``/``str.rjust``. No ``rich``, no ``tabulate`` — a
  fifteen-line formatter is not worth a dependency, and this output has to survive being
  pasted into a terminal, a pull request and a log aggregator unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import psycopg

from durable_llm import __version__
from durable_llm.db import DEFAULT_DSN_ENV, connect, default_dsn, describe_dsn, transaction
from durable_llm.errors import LLMRunError
from durable_llm.migrate import MigrationError, migrate
from durable_llm.providers.base import PROVIDER_NAMES, get_provider
from durable_llm.registry import get as get_pipeline
from durable_llm.registry import load_module
from durable_llm.types import RUN_STATUSES
from durable_llm.worker import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_POLL_INTERVAL,
    default_worker_id,
    run_forever,
    run_once,
    validate_worker_timings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from psycopg.rows import DictRow

    from durable_llm.pipeline import Pipeline

__all__ = ["build_parser", "main"]

logger = logging.getLogger("durable_llm.cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

PROVIDER_ENV = "DURABLE_LLM_PROVIDER"
"""Default for ``--provider``. Mirrors .env.example, so ``make demo`` needs no flags."""

_DEFAULT_LIST_LIMIT = 20
_LABEL_WIDTH = 12
_RULE_WIDTH = 92
_HEX = frozenset("0123456789abcdef-")

_EPILOG = """\
examples:
  durable-llm migrate
  durable-llm enqueue summarise_thread --tenant acme --json '{"thread_id": "t1"}' \\
      --import examples/summarise_thread.py
  durable-llm worker --once --provider echo --import examples/summarise_thread.py
  durable-llm runs show 4a0c8a1e
  durable-llm costs --tenant acme
"""


class _UsageError(Exception):
    """The arguments do not make sense. Exit 2, the same as an argparse rejection."""


class _CommandError(Exception):
    """The arguments were fine; the command could not do what was asked. Exit 1."""


# ======================================================================================
#  SQL   (every statement parameterised; no value is ever formatted into these strings)
# ======================================================================================

_RUNS_LIST_SQL = """
SELECT id, pipeline, tenant_id, status, attempt, max_attempts, failures, spent_usd,
       budget_usd, error_class, locked_by, locked_at, next_attempt_at, created_at, updated_at
FROM runs
WHERE (%s::text IS NULL OR tenant_id = %s::text)
  AND (%s::text IS NULL OR status = %s::text)
ORDER BY created_at DESC
LIMIT %s
"""

_RUN_BY_ID_SQL = "SELECT * FROM runs WHERE id = %s"

# Prefix lookup, so a run id truncated by a terminal still resolves. LIMIT 5 is enough to
# distinguish "no match" from "one match" from "ambiguous".
#
# The LIMIT does not bound the *scan*, which is what the previous version of this comment
# claimed: a LIMIT stops the query once it has enough rows, but a predicate no index can
# serve still has to read the table to find them, and a btree on the uuid cannot serve a
# prefix match on its text rendering. This was a sequential scan with a comment telling the
# next reader not to look. `runs_id_prefix_idx` in migrations/0003 indexes `(id::text)` with
# `text_pattern_ops`, which is what makes an anchored LIKE genuinely indexable.
_RUN_BY_PREFIX_SQL = "SELECT * FROM runs WHERE id::text LIKE %s ORDER BY created_at LIMIT 5"

# One row, four numbers, for `durable-llm health`. `stranded` counts runs that have used every
# allowed execution and are waiting for the worker's sweep to give them a terminal status —
# the population that used to accumulate invisibly.
_HEALTH_SQL = """
SELECT count(*) FILTER (WHERE status = 'queued')                          AS queued,
       count(*) FILTER (WHERE status = 'running')                         AS running,
       count(*) FILTER (WHERE status IN ('queued', 'running')
                          AND attempt >= max_attempts)                    AS stranded,
       (SELECT count(*) FROM schema_migrations)                           AS migrations
FROM runs
"""

_STEPS_SQL = """
SELECT name, status, attempt, error_class, error_detail, started_at, ended_at
FROM steps
WHERE run_id = %s
ORDER BY started_at, name
"""

_CALLS_SQL = """
SELECT step_name, attempt, input_hash, model, tokens_in, tokens_out, cost_usd, cached,
       created_at
FROM llm_calls
WHERE run_id = %s
ORDER BY created_at, step_name, attempt
"""

# Spend by pipeline and model, with the grand total in the same result set.
#
# `billed` and the outer query are BOTH scoped by tenant and time window. They used not to
# be: the CTE read every `cached = false` row in the table regardless of `--tenant`, so
# asking one small tenant what it had spent cost a scan and an external sort of the entire
# lifetime ledger — a report whose cost grew with everyone else's traffic. The ledger is
# append-only by design and nothing prunes it, so that number only ever goes up.
#
# The window is mandatory (the command defaults it to the last 30 days) for the same reason.
# An all-time aggregate over an append-only table is a query that gets slower forever, and it
# should be something you ask for rather than something you get by accident.
#
# `billed` picks the first genuinely-paid-for call of each (run_id, input_hash). Joining
# every row back to it is what lets a replay report what it *would* have cost: a cached row
# stores cost 0 — that is the point of it — so without the join the saving is invisible.
# migrations/0003 makes (run_id, input_hash) WHERE cached = false UNIQUE, so DISTINCT ON now
# picks from a set the database guarantees holds at most one row; it is kept because the
# ORDER BY still has to name a winner for databases migrated before that constraint existed.
#
# The token sums are FILTERed to billed calls for the same reason. A replayed row copies the
# original's token counts (context.py does that deliberately, so a saving can be priced), and
# adding them into the totals would report tokens nobody ever bought next to a spend figure
# that correctly excludes them. TOK IN/TOK OUT are what was *purchased*; SAVED USD is what
# the replays were worth.
#
# GROUPING SETS appends the total as one more row rather than a second round trip, and keeps
# count(DISTINCT run_id) honest at the total level: a run spans several models, so per-group
# run counts cannot simply be added up. grouping() marks which row is which.
_COSTS_SQL = """
WITH scoped AS (
    SELECT c.run_id, c.input_hash, c.model, c.tokens_in, c.tokens_out, c.cost_usd, c.cached,
           r.pipeline
    FROM llm_calls c
    JOIN runs r ON r.id = c.run_id
    WHERE (%s::text IS NULL OR r.tenant_id = %s::text)
      AND c.created_at >= %s
      AND (%s::timestamptz IS NULL OR c.created_at < %s::timestamptz)
),
billed AS (
    SELECT DISTINCT ON (run_id, input_hash) run_id, input_hash, cost_usd
    FROM scoped
    WHERE cached = false
    ORDER BY run_id, input_hash
)
SELECT grouping(s.pipeline, s.model)                         AS is_total,
       coalesce(s.pipeline, 'TOTAL')                         AS pipeline,
       coalesce(s.model, '')                                 AS model,
       count(DISTINCT s.run_id)                              AS runs,
       count(*) FILTER (WHERE NOT s.cached)                  AS billed_calls,
       count(*) FILTER (WHERE s.cached)                      AS cached_calls,
       coalesce(sum(s.tokens_in) FILTER (WHERE NOT s.cached), 0)   AS tokens_in,
       coalesce(sum(s.tokens_out) FILTER (WHERE NOT s.cached), 0)  AS tokens_out,
       coalesce(sum(s.cost_usd), 0)                          AS spend_usd,
       coalesce(sum(b.cost_usd) FILTER (WHERE s.cached), 0)  AS saved_usd
FROM scoped s
LEFT JOIN billed b ON b.run_id = s.run_id AND b.input_hash = s.input_hash
GROUP BY GROUPING SETS ((s.pipeline, s.model), ())
ORDER BY is_total, spend_usd DESC, pipeline, model
"""

DEFAULT_COSTS_WINDOW_DAYS = 30
"""How far back ``durable-llm costs`` looks when ``--since`` is not given.

A default rather than "everything" because the ledger is append-only and unpruned: an
all-time aggregate is a query whose cost grows forever, and the answer people actually want
from a cost report is almost always "recently".
"""


# ======================================================================================
#  table shapes:  columns paired with one alignment character each ('<' left, '>' right)
# ======================================================================================

# LEASE earns its column because it is the difference between the two things a `running`
# row can mean — being worked on right now, or abandoned by a dead worker — and neither
# `status` nor `attempt` distinguishes them. It renders as the worker id for a held lease and
# as the wait for a run backing off, so the common triage question is answered by the list
# itself instead of by a follow-up `runs show` on each candidate.
_RUNS_COLUMNS = (
    "RUN ID",
    "PIPELINE",
    "TENANT",
    "STATUS",
    "ATT",
    "SPENT USD",
    "CREATED (UTC)",
    "LEASE",
    "ERROR",
)
_RUNS_ALIGNS = "<<<<>><<<"

_STEP_COLUMNS = (
    "STEP",
    "STATUS",
    "ATT",
    "DURATION",
    "CALLS",
    "REPLAYS",
    "COST USD",
    "ERROR",
)
_STEP_ALIGNS = "<<>>>>><"

_CALL_COLUMNS = (
    "",
    "#",
    "STEP",
    "ATT",
    "MODEL",
    "TOK IN",
    "TOK OUT",
    "COST USD",
    "BILLING",
    "SAVED USD",
    "AT (UTC)",
)
_CALL_ALIGNS = "<><><>>><><"

_COST_COLUMNS = (
    "PIPELINE",
    "MODEL",
    "RUNS",
    "BILLED",
    "REPLAYED",
    "TOK IN",
    "TOK OUT",
    "SPEND USD",
    "SAVED USD",
)
_COST_ALIGNS = "<<>>>>>>>"


# ======================================================================================
#  output primitives
# ======================================================================================


def _emit(line: str = "") -> None:
    """Write one line of *result* to stdout.

    Every user-facing byte goes through here, which is why the CLI never calls ``print``:
    one funnel makes it obvious that results are on stdout and commentary is on stderr.
    """
    sys.stdout.write(f"{line}\n")


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: str = "",
    indent: str = "  ",
    footer: Sequence[str] | None = None,
) -> list[str]:
    """Render a fixed-width table. ``aligns`` holds one ``<`` or ``>`` per column."""
    columns = len(headers)
    spec = (aligns + "<" * columns)[:columns]

    widths = [0] * columns
    for row in [headers, *rows, *([footer] if footer is not None else [])]:
        for index, cell in enumerate(row[:columns]):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [
            cell.rjust(widths[i]) if spec[i] == ">" else cell.ljust(widths[i])
            for i, cell in enumerate(cells[:columns])
        ]
        return (indent + "  ".join(padded)).rstrip()

    rule = indent + "  ".join("-" * width for width in widths)
    out = [line(headers), rule]
    out.extend(line(row) for row in rows)
    if footer is not None:
        out.extend([rule, line(footer)])
    return out


def _kv(label: str, value: str) -> str:
    return f"  {label.ljust(_LABEL_WIDTH)}{value}"


def _usd(amount: Decimal | int | float | None, places: int = 6) -> str:
    """Format money. ``None`` — an unset budget — is not zero and must not print as one."""
    if amount is None:
        return "-"
    return f"{Decimal(str(amount)):.{places}f}"


def _count(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _ts(value: datetime | None, *, suffix: str = " UTC") -> str:
    if value is None:
        return "-"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S") + suffix


def _clock(value: datetime | None) -> str:
    """Time of day only — enough to read a ledger's ordering without repeating the date."""
    return "-" if value is None else value.astimezone(UTC).strftime("%H:%M:%S")


def _duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "-"
    seconds = (end - start).total_seconds()
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return "-"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 3]}..."


def _json_block(value: Any, *, limit: int = 2000) -> list[str]:
    """Pretty-print a jsonb column, indented and bounded."""
    try:
        text = json.dumps(value, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - jsonb cannot hold such a value
        text = repr(value)
    if len(text) > limit:
        text = f"{text[:limit]}\n... [{len(text) - limit} more characters]"
    return [f"  {line}" for line in text.splitlines()] or ["  (empty)"]


# ======================================================================================
#  shared plumbing
# ======================================================================================


@contextmanager
def _connection(args: argparse.Namespace) -> Iterator[psycopg.Connection[DictRow]]:
    """Open the database, and roll back anything a command left uncommitted."""
    conn = connect(getattr(args, "dsn", None))
    try:
        yield conn
    finally:
        with suppress(psycopg.Error):
            conn.rollback()
        conn.close()


def _import_modules(specs: Sequence[str] | None) -> None:
    """Import the modules that define pipelines, so the registry is populated.

    A worker cannot execute ``summarise_thread`` unless something in *this process* has
    imported the file that decorated it — the registry is a module-level dict, not a table.
    """
    for spec in specs or ():
        logger.debug("importing %s", spec)
        try:
            load_module(spec)
        except Exception as exc:
            raise _CommandError(f"could not import {spec!r}: {type(exc).__name__}: {exc}") from exc


def _resolve_pipeline(name: str) -> Pipeline:
    try:
        return get_pipeline(name)
    except KeyError as exc:
        # registry.get's message already explains --import; strip KeyError's repr quoting.
        raise _UsageError(str(exc).strip("\"'")) from exc


def _lease_summary(run: DictRow) -> str:
    """One cell answering "is anything happening to this run right now?".

    A held lease shows who holds it and for how long; a run waiting out its run-level
    backoff shows the wait. Everything else is blank, because a terminal run has no lease
    and saying so in every row would be noise.
    """
    if run["locked_by"]:
        held = _duration(run["locked_at"], datetime.now(UTC)) if run["locked_at"] else "?"
        return _truncate(f"{run['locked_by']} {held}", 24)
    if run["next_attempt_at"]:
        return f"retry {_duration(datetime.now(UTC), run['next_attempt_at'])}"
    return ""


def _parse_window_bound(raw: str | None, flag: str) -> datetime | None:
    """Parse a ``--since``/``--until`` value into an aware UTC datetime.

    Accepts an ISO-8601 date or timestamp (``2026-08-01``, ``2026-08-01T12:00:00Z``) or a
    plain day count (``7`` meaning "seven days ago"), because the second form is what people
    actually type at a cost report. A naive timestamp is read as UTC rather than as local
    time: the columns are ``timestamptz``, the database owns the clock, and silently applying
    the operator's timezone would make the same command mean different windows on different
    laptops.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    if text.isdigit():
        return datetime.now(UTC) - timedelta(days=int(text))

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _UsageError(
            f"{flag} must be an ISO-8601 date or timestamp (2026-08-01, "
            f"2026-08-01T12:00:00Z) or a number of days ago, got {raw!r}"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _reject_json_constant(token: str) -> Any:
    """Refuse ``NaN``/``Infinity``/``-Infinity`` in ``--json``.

    They are Python's extension to JSON, not JSON, and ``jsonb`` has no way to store them.
    Rejecting at the boundary keeps the failure a usage error instead of a database error.
    """
    raise _UsageError(
        f"--json contains {token}, which is not valid JSON and cannot be stored in "
        "runs.input. Use a string, or null, if you need to represent it."
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Decode ``--json``. ``-`` reads stdin, so a large payload needs no shell quoting."""
    text = sys.stdin.read() if raw.strip() == "-" else raw
    if not text.strip():
        return {}
    try:
        # `parse_constant` is what makes this function's promise true. Python's json module
        # accepts the bare tokens `NaN`, `Infinity` and `-Infinity` by default — they are not
        # in RFC 8259 and `jsonb` will not store them — so they sailed through the one check
        # `--json` gets and died two layers down inside the INSERT, as a raw psycopg error
        # with exit 1 and a leaked `unnamed portal parameter $4` line, where every other
        # malformed `--json` is caught here and exits 2.
        value = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise _UsageError(f"--json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _UsageError(
            "--json must be a JSON object of pipeline keyword arguments, "
            f"got {type(value).__name__}"
        )
    return value


# ======================================================================================
#  commands
# ======================================================================================


def _cmd_migrate(args: argparse.Namespace) -> int:
    with _connection(args) as conn:
        applied = migrate(conn, args.migrations_dir)
    if not applied:
        _emit("schema is up to date; nothing to apply")
        return EXIT_OK
    for version in applied:
        _emit(f"applied {version}")
    _emit(f"{len(applied)} migration(s) applied")
    return EXIT_OK


def _cmd_worker(args: argparse.Namespace) -> int:
    # Checked before anything is imported or connected, because both of these have a value
    # that breaks the system rather than slowing it: `--lease-seconds 0` makes every running
    # run instantly stealable by every other worker, and `--poll-interval 0` turns the idle
    # loop into a busy loop. Neither announces itself — the worker logs `lease=0s poll=0.0s`
    # once and then looks perfectly healthy. The library refuses the same values (see
    # worker.validate_worker_timings); this turns the ValueError into the CLI's own exit 2.
    try:
        validate_worker_timings(
            lease_seconds=args.lease_seconds,
            poll_interval=args.poll_interval,
        )
    except ValueError as exc:
        raise _UsageError(str(exc)) from exc

    _import_modules(args.imports)
    try:
        provider = get_provider(args.provider)
    except ImportError as exc:
        raise _CommandError(
            f"provider {args.provider!r} needs an optional dependency: {exc}. "
            "Install it with `uv sync --extra anthropic`."
        ) from exc
    except ValueError as exc:
        # argparse's `choices` covers the flag, but not the DURABLE_LLM_PROVIDER environment
        # variable it defaults from — so an unknown name can still arrive here, and it is
        # bad usage rather than a runtime failure.
        raise _UsageError(str(exc)) from exc

    worker_id = args.worker_id or default_worker_id()

    with _connection(args) as conn:
        if args.once:
            run_id = run_once(
                conn,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
                provider=provider,
                default_model=args.model,
            )
            if run_id is None:
                # Nothing on stdout: --once is scriptable, and "" means "nothing to do".
                logger.info("no runs are claimable")
            else:
                _emit(str(run_id))
            return EXIT_OK

        run_forever(
            conn,
            provider=provider,
            worker_id=worker_id,
            lease_seconds=args.lease_seconds,
            poll_interval=args.poll_interval,
            default_model=args.model,
            # How to replace the connection when it dies. The loop cannot work this out for
            # itself — psycopg withholds the password from `conn.info.dsn`, deliberately —
            # so the layer that knew the DSN has to hand over a way to reopen it. Without
            # this the loop caught the failure, slept, and retried the same dead handle
            # forever: a live process, logging once a poll, processing nothing, and looking
            # healthy to anything that only checks whether it is running.
            reconnect=lambda: connect(getattr(args, "dsn", None)),
        )
    return EXIT_OK


def _cmd_health(args: argparse.Namespace) -> int:
    """Answer "can this deployment do any work right now?" with an exit code.

    Exists so a container healthcheck, a readiness probe or a deploy gate has something to
    call. The worker is a poll loop with no socket, so before this there was no way to ask
    it anything — a worker wedged against a dead database and a worker quietly draining a
    queue looked identical from outside the process.

    Deliberately checks *dependencies*, not liveness: it opens the database, confirms the
    schema is present, and reports the backlog. A down database makes this exit non-zero,
    which is the right answer for a readiness probe (do not send me work) and is why a
    liveness probe should not be pointed at it — restarting a worker does not fix Postgres.
    """
    try:
        with _connection(args) as conn, transaction(conn) as cur:
            cur.execute(_HEALTH_SQL)
            row = cur.fetchone()
    except psycopg.Error as exc:
        raise _CommandError(f"database is not reachable: {exc}") from exc

    if row is None:  # pragma: no cover - the aggregate always returns exactly one row
        raise _CommandError("health query returned no row")

    _emit(
        f"ok  queued={row['queued']} running={row['running']} "
        f"stranded={row['stranded']} migrations={row['migrations']}"
    )
    if row["stranded"]:
        # Not a failure — the sweep terminates these on its own schedule — but worth saying
        # out loud, because a growing stranded count means runs are burning executions
        # without recording verdicts, which is a symptom rather than a state.
        logger.warning(
            "%d run(s) have used every allowed execution and are awaiting the sweep",
            row["stranded"],
        )
    return EXIT_OK


def _cmd_enqueue(args: argparse.Namespace) -> int:
    payload = _parse_json_object(args.json_input)
    _import_modules(args.imports)
    pipe = _resolve_pipeline(args.pipeline)

    with _connection(args) as conn:
        run_id = pipe.enqueue(tenant_id=args.tenant, conn=conn, **payload)
        # A no-op if enqueue() already committed; essential if it left that to its caller.
        conn.commit()

    logger.info("queued %s for tenant %s", pipe.name, args.tenant)
    _emit(str(run_id))
    return EXIT_OK


def _cmd_runs_list(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise _UsageError("--limit must be at least 1")

    with _connection(args) as conn, conn.cursor() as cur:
        cur.execute(
            _RUNS_LIST_SQL,
            (args.tenant, args.tenant, args.status, args.status, args.limit),
        )
        runs = cur.fetchall()

    if not runs:
        _emit("no runs match")
        return EXIT_OK

    rows = [
        [
            str(run["id"]),
            run["pipeline"],
            run["tenant_id"],
            run["status"],
            f"{run['attempt']}/{run['max_attempts']}",
            _usd(run["spent_usd"]),
            _ts(run["created_at"], suffix=""),
            _lease_summary(run),
            _truncate(run["error_class"], 14),
        ]
        for run in runs
    ]
    for line in _table(_RUNS_COLUMNS, rows, aligns=_RUNS_ALIGNS):
        _emit(line)
    _emit()
    suffix = ", --limit reached" if len(runs) == args.limit else ""
    _emit(f"  {len(runs)} run(s){suffix}")
    return EXIT_OK


def _cmd_runs_show(args: argparse.Namespace) -> int:
    with _connection(args) as conn:
        run = _fetch_run(conn, args.run_id)
        with conn.cursor() as cur:
            cur.execute(_STEPS_SQL, (run["id"],))
            steps = cur.fetchall()
            cur.execute(_CALLS_SQL, (run["id"],))
            calls = cur.fetchall()

    _render_run_header(run, calls)
    _render_payloads(run)
    _render_steps(steps, calls)
    _render_ledger(calls)
    return EXIT_OK


def _cmd_costs(args: argparse.Namespace) -> int:
    since = _parse_window_bound(args.since, "--since")
    until = _parse_window_bound(args.until, "--until")
    if since is None:
        since = datetime.now(UTC) - timedelta(days=DEFAULT_COSTS_WINDOW_DAYS)
    if until is not None and until <= since:
        raise _UsageError(f"--until ({args.until}) must be after --since ({args.since})")

    with _connection(args) as conn, conn.cursor() as cur:
        cur.execute(_COSTS_SQL, (args.tenant, args.tenant, since, until, until))
        result = cur.fetchall()

    # Stated on every report, because a total that silently covers 30 days reads exactly like
    # a total that covers all time, and the difference is the whole invoice.
    window = f"since {since.isoformat(timespec='seconds')}"
    if until is not None:
        window += f" until {until.isoformat(timespec='seconds')}"
    logger.info("costs %s%s", window, f" for tenant {args.tenant}" if args.tenant else "")

    groups = [row for row in result if not row["is_total"]]
    total = next((row for row in result if row["is_total"]), None)

    _emit(f"COSTS  (tenant: {args.tenant or 'all'})")
    _emit("=" * _RULE_WIDTH)
    if not groups:
        _emit("  no llm calls recorded yet")
        return EXIT_OK

    rows = [_cost_cells(row) for row in groups]
    footer = _cost_cells(total) if total is not None else None
    for line in _table(_COST_COLUMNS, rows, aligns=_COST_ALIGNS, footer=footer):
        _emit(line)

    if total is not None and total["cached_calls"]:
        _emit()
        _emit(
            f"  {total['cached_calls']:,} replayed call(s) cost nothing: "
            f"${_usd(total['saved_usd'])} that a retry would have spent again."
        )
    return EXIT_OK


def _cost_cells(row: DictRow) -> list[str]:
    return [
        row["pipeline"],
        row["model"],
        _count(row["runs"]),
        _count(row["billed_calls"]),
        _count(row["cached_calls"]),
        _count(row["tokens_in"]),
        _count(row["tokens_out"]),
        _usd(row["spend_usd"]),
        _usd(row["saved_usd"]),
    ]


# ======================================================================================
#  `runs show` rendering — the demo's money shot
# ======================================================================================


def _fetch_run(conn: psycopg.Connection[DictRow], token: str) -> DictRow:
    """Look a run up by full id, or by an unambiguous prefix of one."""
    parsed: UUID | None = None
    with suppress(ValueError, AttributeError, TypeError):
        parsed = UUID(token)

    with conn.cursor() as cur:
        if parsed is not None:
            cur.execute(_RUN_BY_ID_SQL, (parsed,))
        else:
            prefix = token.strip().lower()
            if not prefix or set(prefix) - _HEX:
                raise _UsageError(f"{token!r} is neither a run id nor a run id prefix")
            # The wildcard is a *parameter*, not string-formatted SQL.
            cur.execute(_RUN_BY_PREFIX_SQL, (f"{prefix}%",))
        matches = cur.fetchall()

    if not matches:
        raise _CommandError(f"no run matches {token!r}")
    if len(matches) > 1:
        found = ", ".join(str(row["id"]) for row in matches)
        raise _CommandError(f"{token!r} is ambiguous; it matches {found}")
    return matches[0]


def _replay_savings(calls: Sequence[DictRow]) -> tuple[dict[str, Decimal], Decimal]:
    """Price the cache hits.

    A ``cached=true`` row stores ``cost_usd = 0``, so the value of the hit lives on the
    billed row that shares its ``input_hash``. Pairing them up is what turns "6 calls were
    cached" into "$0.006110 was not spent a second time".
    """
    billed_cost: dict[str, Decimal] = {}
    for call in calls:
        if not call["cached"]:
            billed_cost.setdefault(call["input_hash"], call["cost_usd"] or Decimal(0))

    saved = Decimal(0)
    for call in calls:
        if call["cached"]:
            saved += billed_cost.get(call["input_hash"], Decimal(0))
    return billed_cost, saved


def _render_run_header(run: DictRow, calls: Sequence[DictRow]) -> None:
    _, saved = _replay_savings(calls)
    replays = sum(1 for call in calls if call["cached"])
    spent: Decimal = run["spent_usd"] or Decimal(0)
    budget: Decimal | None = run["budget_usd"]

    _emit(f"RUN {run['id']}")
    _emit("=" * _RULE_WIDTH)
    _emit(_kv("pipeline", run["pipeline"]))
    _emit(_kv("tenant", run["tenant_id"]))
    _emit(_kv("status", str(run["status"]).upper()))
    _emit(_kv("attempts", f"{run['attempt']} of {run['max_attempts']} allowed"))
    _emit(_kv("failures", f"{run['failures']} recorded"))

    # The lease, which used to be fetched (`SELECT *`) and then never shown. It is the first
    # thing anyone needs when a run is stuck: *which* box and process is holding it, and for
    # how long. Without it the answer to "is this run being worked on or abandoned?" was a
    # psql session, on the one command whose whole job is to explain a run.
    if run["locked_by"]:
        held = _duration(run["locked_at"], datetime.now(UTC)) if run["locked_at"] else "?"
        _emit(_kv("leased to", f"{run['locked_by']} (held {held})"))
    if run["next_attempt_at"]:
        _emit(_kv("retry at", _ts(run["next_attempt_at"])))

    _emit(_kv("created", _ts(run["created_at"])))
    _emit(_kv("updated", _ts(run["updated_at"])))
    _emit(_kv("wall clock", _duration(run["created_at"], run["updated_at"])))

    if budget is None:
        _emit(_kv("budget", "unlimited"))
        _emit(_kv("spent", f"${_usd(spent)}"))
    else:
        _emit(_kv("budget", f"${_usd(budget)}"))
        share = f"  ({spent / budget * 100:.1f}% of budget)" if budget > 0 else ""
        _emit(_kv("spent", f"${_usd(spent)}{share}"))

    if calls:
        detail = f"{replays} of {len(calls)} calls replayed"
        if replays:
            detail = f"{detail}, ${_usd(saved)} not spent again"
        _emit(_kv("cache", detail))

    if run["error_class"]:
        _emit(_kv("error", f"{run['error_class']}  {_truncate(run['error_detail'], 100)}"))


def _render_payloads(run: DictRow) -> None:
    _emit()
    _emit("INPUT")
    for line in _json_block(run["input"]):
        _emit(line)

    if run["output"] is not None:
        _emit()
        _emit("OUTPUT")
        for line in _json_block(run["output"]):
            _emit(line)


def _render_steps(steps: Sequence[DictRow], calls: Sequence[DictRow]) -> None:
    _emit()
    _emit("STEPS")
    if not steps:
        _emit("  no steps recorded yet")
        return

    billed = Counter(call["step_name"] for call in calls if not call["cached"])
    replayed = Counter(call["step_name"] for call in calls if call["cached"])
    cost: defaultdict[str, Decimal] = defaultdict(Decimal)
    for call in calls:
        cost[call["step_name"]] += call["cost_usd"] or Decimal(0)

    rows = [
        [
            step["name"],
            step["status"],
            str(step["attempt"]),
            _duration(step["started_at"], step["ended_at"]),
            str(billed[step["name"]]),
            str(replayed[step["name"]]),
            _usd(cost[step["name"]]),
            _truncate(step["error_class"], 16),
        ]
        for step in steps
    ]
    for line in _table(_STEP_COLUMNS, rows, aligns=_STEP_ALIGNS):
        _emit(line)

    for step in steps:
        if step["error_detail"]:
            _emit(f"  ! {step['name']}: {_truncate(step['error_detail'], 110)}")


def _render_ledger(calls: Sequence[DictRow]) -> None:
    """Print the per-call cost ledger, with replayed calls impossible to miss.

    Three independent signals mark a replay — the ``*`` gutter, the ``CACHED`` billing
    column, and a non-zero ``SAVED USD`` figure — because this table is the evidence for
    the project's central claim, and evidence that needs explaining is not evidence.
    """
    _emit()
    _emit("LLM CALLS  (cost ledger)")
    if not calls:
        _emit("  no provider calls recorded yet")
        return

    billed_cost, saved = _replay_savings(calls)
    rows: list[list[str]] = []
    for index, call in enumerate(calls, start=1):
        cached = bool(call["cached"])
        rows.append(
            [
                "*" if cached else "",
                str(index),
                call["step_name"],
                str(call["attempt"]),
                call["model"],
                _count(call["tokens_in"]),
                _count(call["tokens_out"]),
                _usd(call["cost_usd"] or Decimal(0)),
                "CACHED" if cached else "billed",
                _usd(billed_cost.get(call["input_hash"], Decimal(0))) if cached else "",
                _clock(call["created_at"]),
            ]
        )

    for line in _table(_CALL_COLUMNS, rows, aligns=_CALL_ALIGNS):
        _emit(line)

    replays = sum(1 for call in calls if call["cached"])
    _emit()
    if replays:
        _emit(
            f"  * {replays} of {len(calls)} calls were replayed from this run's ledger: "
            "billed $0.000000, where"
        )
        _emit(
            f"    an unguarded retry would have spent ${_usd(saved)} re-buying tokens it "
            "already had."
        )
    else:
        _emit("  no replays: every call in this run was billed exactly once.")


# ======================================================================================
#  argument parsing
# ======================================================================================


def _global_options() -> argparse.ArgumentParser:
    """Flags accepted both before and after the subcommand.

    ``argparse.SUPPRESS`` is load-bearing. A subparser writes its own defaults into the
    same namespace *after* the top-level parser has filled it, so with an ordinary default
    ``durable-llm -v worker`` would parse the ``-v`` and then silently overwrite it with ``0``.
    Suppressed defaults leave the attribute absent instead, which is why every read below
    goes through ``getattr(args, ..., fallback)``.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--dsn",
        default=argparse.SUPPRESS,
        metavar="DSN",
        help=f"Postgres connection string (default: ${DEFAULT_DSN_ENV}, else {default_dsn()})",
    )
    shared.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="log at DEBUG",
    )
    shared.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="log warnings and worse only",
    )
    return shared


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser. Separate from :func:`main` so tests can inspect it."""
    shared = _global_options()
    parser = argparse.ArgumentParser(
        prog="durable-llm",
        parents=[shared],
        description=(
            "Run multi-step LLM pipelines that survive crashes, do not double-bill on "
            "retry, and cannot silently blow a budget. All state lives in Postgres."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"durable-llm {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    migrate_parser = sub.add_parser(
        "migrate",
        parents=[shared],
        help="apply pending SQL migrations",
        description="Apply migrations/*.sql in filename order. Idempotent.",
    )
    migrate_parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory of .sql files (default: the bundled migrations/)",
    )
    migrate_parser.set_defaults(func=_cmd_migrate)

    worker_parser = sub.add_parser(
        "worker",
        parents=[shared],
        help="claim queued runs and execute them",
        description=(
            "Claim runs with a lease and execute them. Runs until SIGTERM or SIGINT, "
            "finishing the run in flight before exiting."
        ),
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one run, then exit (prints its id, or nothing when idle)",
    )
    worker_parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="module or .py path defining pipelines; repeatable",
    )
    worker_parser.add_argument(
        "--provider",
        # `choices` makes providers/base.py's stated invariant true and routes a typo
        # through argparse's exit 2, matching this CLI's own contract for bad usage. It
        # does NOT validate the environment default below, which argparse never checks —
        # `_cmd_worker` catches the ValueError from get_provider for that path.
        choices=sorted(PROVIDER_NAMES),
        default=os.environ.get(PROVIDER_ENV, "echo").strip() or "echo",
        metavar="NAME",
        help=f"one of {', '.join(PROVIDER_NAMES)} (default: ${PROVIDER_ENV}, else echo)",
    )
    worker_parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
        metavar="N",
        help="how long before a silent worker's run may be reclaimed (default: %(default)s)",
    )
    worker_parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        metavar="SECONDS",
        help="idle wait between claim attempts (default: %(default)s)",
    )
    worker_parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="default model for ctx.llm() calls that do not name one",
    )
    worker_parser.add_argument(
        "--worker-id",
        default=None,
        metavar="ID",
        help="identity written to runs.locked_by (default: hostname/pid)",
    )
    worker_parser.set_defaults(func=_cmd_worker)

    enqueue_parser = sub.add_parser(
        "enqueue",
        parents=[shared],
        help="put a run on the queue",
        description="Insert a queued run and print its id. Executes nothing.",
    )
    enqueue_parser.add_argument("pipeline", metavar="PIPELINE", help="registered pipeline name")
    enqueue_parser.add_argument("--tenant", required=True, metavar="T", help="tenant id")
    enqueue_parser.add_argument(
        "--json",
        dest="json_input",
        default="{}",
        metavar="JSON",
        help="pipeline keyword arguments as a JSON object ('-' reads stdin)",
    )
    enqueue_parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="module or .py path defining pipelines; repeatable",
    )
    enqueue_parser.set_defaults(func=_cmd_enqueue)

    runs_parser = sub.add_parser("runs", parents=[shared], help="inspect runs")
    runs_sub = runs_parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND", required=True)

    list_parser = runs_sub.add_parser("list", parents=[shared], help="most recent runs")
    list_parser.add_argument("--tenant", default=None, metavar="T")
    list_parser.add_argument("--status", default=None, choices=RUN_STATUSES, metavar="S")
    list_parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIST_LIMIT,
        metavar="N",
        help="default: %(default)s",
    )
    list_parser.set_defaults(func=_cmd_runs_list)

    show_parser = runs_sub.add_parser(
        "show",
        parents=[shared],
        help="one run in full: header, steps, cost ledger",
        description="Print a run's header, its step timeline and its per-call cost ledger.",
    )
    show_parser.add_argument("run_id", metavar="RUN_ID", help="full run id, or a prefix of one")
    show_parser.set_defaults(func=_cmd_runs_show)

    costs_parser = sub.add_parser(
        "costs",
        parents=[shared],
        help="spend by pipeline and model",
        description="Aggregate spend, token counts and cache savings.",
    )
    costs_parser.add_argument("--tenant", default=None, metavar="T")
    costs_parser.add_argument(
        "--since",
        default=None,
        metavar="WHEN",
        help=(
            "ISO-8601 date/timestamp, or a number of days ago "
            f"(default: {DEFAULT_COSTS_WINDOW_DAYS} days)"
        ),
    )
    costs_parser.add_argument(
        "--until",
        default=None,
        metavar="WHEN",
        help="exclusive upper bound, same formats as --since (default: no upper bound)",
    )
    costs_parser.set_defaults(func=_cmd_costs)

    health_parser = sub.add_parser(
        "health",
        parents=[shared],
        help="check the database is reachable and report the backlog",
        description=(
            "Dependency check for a container healthcheck or readiness probe. Exits "
            "non-zero when the database cannot be reached."
        ),
    )
    health_parser.set_defaults(func=_cmd_health)

    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    """Send diagnostics to stderr, so stdout stays a clean, pipeable result.

    The timestamp is a full ISO-8601 instant in UTC, not a bare local ``%H:%M:%S``. A worker
    is a long-lived process whose logs are read days later, next to logs from other hosts:
    a time with no date cannot be correlated with a ``created_at`` in the database, and a
    local time with no offset cannot be correlated with anything on a differently-configured
    machine. The process id is included for the same reason — several workers per host is
    the expected deployment, and ``runs.locked_by`` identifies them as ``hostname/pid``.
    """
    if getattr(args, "verbose", 0):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ %(levelname)-7s pid=%(process)d %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # UTC rather than local time, set on the formatter because logging's converter is a
    # module-level default that a library must not mutate for its host application.
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler], force=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for an expected failure."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits directly for --help (0), --version (0) and usage errors (2).
        # main() promises to return a code rather than raise, so translate it back.
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    _configure_logging(args)

    try:
        return args.func(args)
    except _UsageError as exc:
        logger.error("%s", exc)
        logger.error("run `durable-llm %s --help` for usage", args.command)
        return EXIT_USAGE
    except BrokenPipeError:
        # `durable-llm runs show ... | head` closes stdout early. Not an error — but the
        # interpreter must not try to flush the dead pipe on the way out.
        with suppress(OSError):
            sys.stdout.close()
        return EXIT_OK
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_ERROR
    except _CommandError as exc:
        logger.error("%s", exc)
        logger.debug("traceback", exc_info=exc)
        return EXIT_ERROR
    except psycopg.OperationalError as exc:
        logger.error(
            "cannot reach Postgres at %s: %s",
            describe_dsn(getattr(args, "dsn", None) or default_dsn()),
            _truncate(str(exc), 160),
        )
        logger.error("start it with `make up`, or point $%s at your database", DEFAULT_DSN_ENV)
        return EXIT_ERROR
    except (LLMRunError, MigrationError, psycopg.Error, ValueError, KeyError, OSError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        logger.debug("traceback", exc_info=exc)
        return EXIT_ERROR
    except Exception:
        # Last resort: a bug in durable_llm itself. Show the traceback — an unexplained exit
        # code helps nobody.
        logger.exception("unexpected error")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
