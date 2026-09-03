"""Shared fixtures. Every test in this suite runs against a real Postgres.

That is a deliberate choice, not laziness about mocking. Everything durable_llm claims —
a crashed worker resumes without re-billing, a budget cannot be exceeded by two workers
racing, a step is either committed or it never happened — is a property of *Postgres
transactions*. A fake database would let all four assertions pass while the real system
was broken, which is worse than having no tests.

So the suite needs a database, and the one thing it must never do is greet a new
contributor with a psycopg connection traceback. If Postgres is unreachable the whole
suite skips with :data:`_SKIP_HINT`, which tells them to run ``make up``.

Layout:

* :func:`database_url` / :func:`_migrated_database` — session-scoped: resolve the DSN,
  connect once, apply migrations once.
* :func:`conn` — a fresh connection per test, so a test that leaves a transaction in a
  failed state cannot poison the next one.
* :func:`db` — small query helpers, so tests read as assertions instead of as SQL.
* :func:`_clean_tables` — autouse: truncates the three mutable tables before every test.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
import pytest
from psycopg.rows import DictRow

from durable_llm.db import connect, default_dsn, describe_dsn, transaction
from durable_llm.migrate import MigrationError, migrate
from durable_llm.types import RunRecord, RunStatus, StepRecord

logger = logging.getLogger("durable_llm.tests")

# --------------------------------------------------------------------------------------
# SQL. Every statement here is a module-level literal with %s placeholders: no f-strings,
# no string building from test input, not even in test code.
# --------------------------------------------------------------------------------------

#: ``runs`` is truncated last only for readability — CASCADE means the order is irrelevant.
#: ``schema_migrations`` is deliberately absent: the schema is applied once per session,
#: and re-migrating between tests would be slow and would prove nothing.
_TRUNCATE = "TRUNCATE TABLE llm_calls, steps, runs RESTART IDENTITY CASCADE"

_INSERT_RUN = """
    INSERT INTO runs (pipeline, tenant_id, status, input, budget_usd, spent_usd)
    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
    RETURNING id
"""

_SELECT_RUN = "SELECT * FROM runs WHERE id = %s::uuid"

_SELECT_LATEST_RUN = """
    SELECT id FROM runs
    WHERE tenant_id = %s
    ORDER BY created_at DESC, id
    LIMIT 1
"""

#: Puts a run back on the queue and drops the lease. This is precisely what the worker's
#: claim query does to a run abandoned by a killed worker, minus the wait for the lease to
#: expire — which is why tests can use it to force a resumption without sleeping.
_REQUEUE = """
    UPDATE runs
    SET status = 'queued',
        output = NULL,
        error_class = NULL,
        error_detail = NULL,
        locked_by = NULL,
        locked_at = NULL,
        updated_at = now()
    WHERE id = %s::uuid
"""

_REQUEUE_WITH_INPUT = """
    UPDATE runs
    SET status = 'queued',
        input = %s::jsonb,
        output = NULL,
        error_class = NULL,
        error_detail = NULL,
        locked_by = NULL,
        locked_at = NULL,
        updated_at = now()
    WHERE id = %s::uuid
"""

_SELECT_STEPS = "SELECT * FROM steps WHERE run_id = %s::uuid ORDER BY started_at, name"

_COUNT_STEPS = "SELECT count(*) AS n FROM steps WHERE run_id = %s::uuid"
_COUNT_STEPS_BY_STATUS = _COUNT_STEPS + " AND status = %s"

_COUNT_CALLS = "SELECT count(*) AS n FROM llm_calls WHERE run_id = %s::uuid"
_COUNT_CALLS_BY_CACHED = _COUNT_CALLS + " AND cached = %s"

_SELECT_SPEND = "SELECT spent_usd FROM runs WHERE id = %s::uuid"

#: Both counters in ONE statement, so they come from a single MVCC snapshot. Read as two
#: queries they could straddle a worker's commit and report a state that never existed —
#: which is exactly the state the chaos suite is watching for, so a false positive there
#: would fire the SIGKILL at the wrong instant.
_SELECT_PROGRESS = """
    SELECT (SELECT count(*) FROM steps
             WHERE run_id = %s::uuid AND status = 'succeeded')      AS succeeded_steps,
           (SELECT count(*) FROM llm_calls
             WHERE run_id = %s::uuid AND cached = false)            AS billed_calls
"""

_SKIP_HINT = (
    "durable_llm's tests need a real Postgres — the crash-safety guarantees under test are "
    "properties of Postgres transactions, so there is nothing to assert against a mock.\n"
    "Start one with:  make up\n"
    "Or point DURABLE_LLM_DATABASE_URL at your own Postgres 17."
)


# --------------------------------------------------------------------------------------
# session-scoped database
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def database_url() -> str:
    """The DSN under test: ``$DURABLE_LLM_DATABASE_URL``, or the docker-compose default."""
    return default_dsn()


@pytest.fixture(scope="session")
def _migrated_database(database_url: str) -> Iterator[str]:
    """Connect once and bring the schema up to date, or skip the suite with advice.

    ``connect_timeout`` is short because the failure being guarded against — "nobody ran
    ``make up``" — is instant, and a developer should not wait out a TCP timeout to be told
    so. A genuinely slow database will still connect: the timeout applies to establishing
    the connection, not to the queries that follow.
    """
    try:
        conn = connect(database_url, connect_timeout=5)
    # Broad on purpose: an unreachable host, a missing database and a malformed DSN all
    # deserve the same actionable message rather than a driver traceback.
    except psycopg.Error as exc:
        pytest.skip(
            f"cannot reach Postgres at {describe_dsn(database_url)}.\n"
            f"{_SKIP_HINT}\n"
            f"({type(exc).__name__}: {str(exc).strip()})"
        )

    try:
        applied = migrate(conn)
    except (MigrationError, psycopg.Error) as exc:
        conn.close()
        pytest.skip(
            f"Postgres at {describe_dsn(database_url)} answered but migrations failed.\n"
            f"{_SKIP_HINT}\n"
            f"({type(exc).__name__}: {str(exc).strip()})"
        )

    # Logged rather than printed: pytest surfaces captured logs on failure, which is the
    # only time "the suite migrated your database" is information anyone wants.
    logger.info(
        "test database %s ready (applied %d migration(s))",
        describe_dsn(database_url),
        len(applied),
    )

    try:
        yield database_url
    finally:
        conn.close()


@pytest.fixture
def conn(_migrated_database: str) -> Iterator[psycopg.Connection[DictRow]]:
    """A fresh connection per test.

    Per-test rather than per-session because ``autocommit=False`` means one test that
    leaves an aborted transaction open would make every following test fail with
    "current transaction is aborted" — a debugging experience nobody deserves.
    """
    connection = connect(_migrated_database)
    try:
        yield connection
    finally:
        connection.close()


# --------------------------------------------------------------------------------------
# query helpers
# --------------------------------------------------------------------------------------


class Database:
    """Query helpers over the four tables, so tests assert on facts rather than on SQL.

    Every method runs inside :func:`durable_llm.db.transaction`, i.e. commits before returning.
    That matters for the chaos suite: its polling loops must observe rows committed by a
    *worker subprocess*, and a long-lived read transaction on this connection would keep
    handing back a stale snapshot.
    """

    def __init__(self, conn: psycopg.Connection[DictRow]) -> None:
        self.conn = conn

    # -- mutation ------------------------------------------------------------------

    def truncate(self) -> None:
        """Empty the three mutable tables, leaving the schema in place."""
        with transaction(self.conn) as cur:
            cur.execute(_TRUNCATE)

    def insert_run(
        self,
        *,
        pipeline: str = "fixture_pipeline",
        tenant_id: str = "fixture",
        status: RunStatus = "queued",
        run_input: dict[str, Any] | None = None,
        budget_usd: Decimal | None = None,
        spent_usd: Decimal = Decimal("0"),
    ) -> UUID:
        """Insert a ``runs`` row directly.

        Used by tests that exercise a helper (``check_budget``, ``record_spend``) against a
        row in a specific state, without going through a pipeline to get there.
        """
        with transaction(self.conn) as cur:
            cur.execute(
                _INSERT_RUN,
                (
                    pipeline,
                    tenant_id,
                    status,
                    json.dumps(run_input or {}),
                    budget_usd,
                    spent_usd,
                ),
            )
            row = cur.fetchone()
        assert row is not None, "INSERT ... RETURNING id returned no row"
        return row["id"]

    def clear_backoff(self, run_id: UUID) -> None:
        """Make a requeued run immediately claimable again.

        ``_record_failure`` sets ``runs.next_attempt_at`` to a jittered backoff so a failing
        run is not re-claimed on the very next poll (migrations/0003). That is correct in
        production and merely slow in a test that is asserting something else — so a test
        about the *ceiling* or about *replay* clears the wait explicitly rather than
        sleeping through it, and the pacing gets its own test instead of being smuggled into
        every other one.
        """
        with transaction(self.conn) as cur:
            cur.execute(
                "UPDATE runs SET next_attempt_at = NULL WHERE id = %s::uuid", (str(run_id),)
            )

    def next_attempt_at(self, run_id: UUID) -> datetime | None:
        """When this run becomes claimable again, or ``None`` for "now"."""
        with transaction(self.conn) as cur:
            cur.execute("SELECT next_attempt_at FROM runs WHERE id = %s::uuid", (str(run_id),))
            row = cur.fetchone()
        assert row is not None, f"run {run_id} does not exist"
        value: datetime | None = row["next_attempt_at"]
        return value

    def requeue(self, run_id: UUID, *, run_input: dict[str, Any] | None = None) -> None:
        """Return a finished run to the queue, optionally with different input.

        Clearing ``output`` is the point: a resumption test that left the old output in
        place could not tell "replayed from checkpoints" apart from "nothing happened".
        """
        with transaction(self.conn) as cur:
            if run_input is None:
                cur.execute(_REQUEUE, (str(run_id),))
            else:
                cur.execute(_REQUEUE_WITH_INPUT, (json.dumps(run_input), str(run_id)))

    # -- reads ---------------------------------------------------------------------

    def run(self, run_id: UUID) -> RunRecord:
        """The run, as a typed record. Fails the test if it does not exist."""
        record = self.maybe_run(run_id)
        if record is None:
            pytest.fail(f"run {run_id} does not exist")
        return record

    def maybe_run(self, run_id: UUID) -> RunRecord | None:
        with transaction(self.conn) as cur:
            cur.execute(_SELECT_RUN, (str(run_id),))
            row = cur.fetchone()
        return None if row is None else RunRecord.from_row(row)

    def latest_run_id(self, tenant_id: str) -> UUID:
        """The newest run for a tenant — how the chaos suite finds the run the CLI queued.

        Reading the id back from the database rather than parsing it out of the CLI's
        stdout keeps these tests decoupled from how ``durable-llm enqueue`` formats its output.
        """
        with transaction(self.conn) as cur:
            cur.execute(_SELECT_LATEST_RUN, (tenant_id,))
            row = cur.fetchone()
        if row is None:
            pytest.fail(f"no run exists for tenant {tenant_id!r}")
        return row["id"]

    def steps(self, run_id: UUID) -> list[StepRecord]:
        with transaction(self.conn) as cur:
            cur.execute(_SELECT_STEPS, (str(run_id),))
            rows = cur.fetchall()
        return [StepRecord.from_row(row) for row in rows]

    def count_steps(self, run_id: UUID, *, status: str | None = None) -> int:
        with transaction(self.conn) as cur:
            if status is None:
                cur.execute(_COUNT_STEPS, (str(run_id),))
            else:
                cur.execute(_COUNT_STEPS_BY_STATUS, (str(run_id), status))
            row = cur.fetchone()
        assert row is not None
        return int(row["n"])

    def count_llm_calls(self, run_id: UUID, *, cached: bool | None = None) -> int:
        """Count ledger rows. ``cached=False`` counts *billed* calls — the money number."""
        with transaction(self.conn) as cur:
            if cached is None:
                cur.execute(_COUNT_CALLS, (str(run_id),))
            else:
                cur.execute(_COUNT_CALLS_BY_CACHED, (str(run_id), cached))
            row = cur.fetchone()
        assert row is not None
        return int(row["n"])

    def progress(self, run_id: UUID) -> tuple[int, int]:
        """``(succeeded_steps, billed_calls)`` as of one consistent snapshot.

        The pair is what tells the chaos suite *where inside a step* the worker currently
        is. Between steps the two settle into a fixed relationship (every billed call has a
        committed step behind it); while a step is mid-flight — its ledger row committed but
        its checkpoint not yet written — the call count runs one ahead. That transient
        inequality is the only externally visible signal of the window the ``llm_calls``
        idempotency store exists to cover.
        """
        with transaction(self.conn) as cur:
            cur.execute(_SELECT_PROGRESS, (str(run_id), str(run_id)))
            row = cur.fetchone()
        assert row is not None
        return int(row["succeeded_steps"]), int(row["billed_calls"])

    def spent(self, run_id: UUID) -> Decimal:
        with transaction(self.conn) as cur:
            cur.execute(_SELECT_SPEND, (str(run_id),))
            row = cur.fetchone()
        if row is None:
            pytest.fail(f"run {run_id} does not exist")
        value = row["spent_usd"]
        # Never Decimal(float): psycopg returns Decimal already, and the coercion below is
        # only reached if a differently-configured cursor ever hands back a string.
        return value if isinstance(value, Decimal) else Decimal(str(value))


@pytest.fixture
def db(conn: psycopg.Connection[DictRow]) -> Database:
    """Query helpers bound to this test's connection."""
    return Database(conn)


@pytest.fixture
def reset_tables(db: Database) -> Callable[[], None]:
    """Truncate on demand — used by the chaos suite between its baseline and crash phases."""
    return db.truncate


@pytest.fixture(autouse=True)
def _clean_tables(reset_tables: Callable[[], None]) -> None:
    """Start every test from an empty ``runs``/``steps``/``llm_calls``.

    Autouse, and therefore the reason the *whole* suite skips rather than errors when
    Postgres is down: it drags in ``conn`` and the session fixture behind it for every
    test, so the skip message is delivered exactly once per test instead of a stack trace.

    Truncate rather than a rollback-per-test wrapper because the chaos suite's worker
    subprocesses run in their own connections and must see committed rows.
    """
    reset_tables()
