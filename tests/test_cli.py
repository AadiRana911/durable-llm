"""The user-facing surface: ~1000 lines of ``cli.py`` that nothing else in the suite touches.

Everything llmrun does correctly is invisible unless a command prints it. ``runs show`` is
the evidence for the project's central claim — that a crashed run replays instead of
re-buying tokens — and ``costs`` is the number somebody will put in a budget review. Both
were entirely untested, which meant the engine's guarantees were verified and the only way
anyone observes them was not.

Everything here drives :func:`llmrun.cli.main` **in this process**, against the same Postgres
the rest of the suite uses, with ``capsys`` for the output. No subprocesses: a CLI test suite
that costs a second per case gets skipped locally, then trimmed, then deleted. In-process
also buys a property a subprocess cannot — the pipeline under test is registered by *this*
module's ``@pipeline`` decorator, so ``enqueue`` and ``worker`` need no ``--import`` and the
tests exercise the commands rather than the module loader (which ``tests/test_registry.py``
owns).

Three conventions this file leans on, all of them contracts ``cli.py`` states in its own
docstring and none of which anything checked:

* **Exit codes mean something.** ``0`` worked, ``1`` the command failed, ``2`` the arguments
  were wrong. The difference between 1 and 2 is the difference between "your database is
  down" and "you typed it wrong", and several tests here exist only to pin which side of that
  line a given input falls on.
* **stdout is the result; stderr is commentary.** ``run_id=$(llmrun enqueue ...)`` is a
  documented usage, so a diagnostic leaking onto stdout is a broken pipeline for somebody.
  Every failure case below asserts stdout is *empty*, not merely that the code is non-zero.
* **Tables are the API.** The fixed-width layout is what people read, paste and grep. The
  parser below reads columns back out of the rule line, so a test can assert on a *cell*
  rather than on a substring that would also match a different column's contents.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID

import pytest

from llmrun.cli import _LABEL_WIDTH, main
from llmrun.context import RunContext
from llmrun.db import transaction
from llmrun.pipeline import pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from conftest import Database

# --------------------------------------------------------------------------------------
# SQL. Module-level literals with %s placeholders, as in conftest: no f-strings, no string
# building from test input, not even here.
# --------------------------------------------------------------------------------------

#: Every column a test in this file ever needs to control, always passed, so there is one
#: statement rather than a family of them assembled at runtime. ``coalesce`` on the id lets a
#: caller pin a uuid (the prefix-collision tests need two ids sharing a leading string, which
#: ``gen_random_uuid`` will not produce this century) or leave it to the database.
_INSERT_RUN = """
INSERT INTO runs (id, pipeline, tenant_id, status, input, spent_usd, attempt, failures,
                  max_attempts, error_class, locked_by, locked_at, next_attempt_at)
VALUES (coalesce(%s::uuid, gen_random_uuid()), %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s,
        %s, CASE WHEN %s::text IS NULL THEN NULL ELSE now() END,
        now() + make_interval(secs => %s))
RETURNING id
"""

#: A ledger row placed at a chosen age, so the ``costs`` window can be tested without
#: waiting thirty days. ``created_at`` is normally defaulted by the database; overriding it
#: is the only way to put a row outside the report's default window.
_INSERT_CALL = """
INSERT INTO llm_calls (run_id, step_name, attempt, input_hash, model, tokens_in, tokens_out,
                       cost_usd, cached, created_at)
VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, now() - make_interval(secs => %s))
"""

#: Simulates the crash window the ``llm_calls`` ledger exists to cover: the provider call is
#: committed, the step checkpoint is not. Deleting the checkpoints of a finished run puts it
#: in exactly that state for every step at once, which is what makes a replay observable in
#: ``runs show`` — with the checkpoints intact, a resumed run never consults the ledger at all.
_DELETE_STEPS = "DELETE FROM steps WHERE run_id = %s::uuid"

_COUNT_MIGRATIONS = "SELECT count(*) AS n FROM schema_migrations"

#: A model with a non-zero price. ``echo`` is deliberately priced at zero, so a run on the
#: default model produces a ledger full of $0.000000 — true, and useless for asserting that
#: a replay *saved* anything. Pointing the offline provider at a priced model is the
#: supported way to make the fake accrue spend (see EchoProvider's docstring).
PRICED_MODEL = "claude-haiku-4-5"

TENANT = "cli-tests"


# --------------------------------------------------------------------------------------
# the pipeline under test
# --------------------------------------------------------------------------------------


def _gather(topic: str) -> dict[str, Any]:
    """The non-LLM step: deterministic, so a resumed run reproduces the same digest."""
    return {"topic": topic, "lines": [f"{topic}-{index}" for index in range(3)]}


@pipeline(name="cli_probe", budget_usd=None, max_attempts=3)
def cli_probe(ctx: RunContext, topic: str) -> dict[str, Any]:
    """One free step and one billed call — the smallest run ``runs show`` can say anything about.

    Kept to a single provider call on purpose. The replay assertions below name exact counts
    ("1 of 2 calls replayed"), and a twelve-step pipeline would express the same property in
    numbers nobody can verify by reading the test.
    """
    gathered = ctx.step("gather", _gather, topic)
    return {"answer": ctx.llm("answer", "[answer] summarise the payload", gathered)}


# --------------------------------------------------------------------------------------
# driving the CLI
# --------------------------------------------------------------------------------------


class CliResult(NamedTuple):
    """One in-process invocation of :func:`llmrun.cli.main`."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def lines(self) -> list[str]:
        return self.stdout.splitlines()


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Undo ``logging.basicConfig(force=True)``, which every ``main()`` call performs.

    ``force=True`` removes *and closes* whatever handlers the root logger had — including the
    one pytest's logging plugin installed to capture records for this test. That is correct
    behaviour for a program that owns its process and unacceptable leakage inside a test
    session, so the handler list and level are snapshotted and put back. Without this, the
    first CLI test in a run silently disarms log capture for every test that follows it in
    the same session, in this file and in everyone else's.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture
def run_cli(database_url: str, capsys: pytest.CaptureFixture[str]) -> Callable[..., CliResult]:
    """Call ``llmrun <args>`` in-process and hand back its exit code and streams.

    ``--dsn`` is prepended rather than left to ``$LLMRUN_DATABASE_URL`` so the tests state
    which database they mean instead of inheriting it from the shell — and it goes *before*
    the subcommand deliberately, which only works because ``cli._global_options`` defaults
    every shared flag to ``argparse.SUPPRESS``. An ordinary default would let the subparser
    overwrite the value the top-level parser just read, and every test in this file would
    quietly run against the compose database instead.
    """

    def _run(*argv: str) -> CliResult:
        exit_code = main(["--dsn", database_url, *argv])
        captured = capsys.readouterr()
        return CliResult(exit_code, captured.out, captured.err)

    return _run


# --------------------------------------------------------------------------------------
# reading the tables back
# --------------------------------------------------------------------------------------


def _column_spans(rule: str) -> list[tuple[int, int]]:
    """Turn ``  ----  ---  -----`` into the character span of each column.

    The rule line is generated from the same widths as the data rows, so it is the one place
    the layout is unambiguous — splitting a row on whitespace would fail the moment a cell is
    empty (the LEASE column of a terminal run) or contains a space (``box/1 4.20s``), which
    are precisely the cells these tests are about.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, char in enumerate(rule):
        if char == "-" and start is None:
            start = index
        elif char != "-" and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(rule)))
    return spans


def _is_rule(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", " "}


def parse_table(lines: Sequence[str], *, after: str = "") -> list[dict[str, str]]:
    """Read a ``cli._table`` block back into one dict per row, keyed by header cell.

    Args:
        after: Substring of a line that precedes the wanted table. ``runs show`` prints three
            tables, so "the first rule line in the output" is not a useful address.

    Rows are collected until the first blank line. Further rule lines are skipped rather than
    treated as data, which is what lets this read the ``costs`` footer — a second rule
    followed by the TOTAL row — as just another row.
    """
    start = 0
    if after:
        start = next(index for index, line in enumerate(lines) if after in line)
    rule_index = next(index for index in range(start, len(lines)) if _is_rule(lines[index]))

    spans = _column_spans(lines[rule_index])
    headers = [lines[rule_index - 1][begin:end].strip() for begin, end in spans]

    rows: list[dict[str, str]] = []
    for line in lines[rule_index + 1 :]:
        if not line.strip():
            break
        if _is_rule(line):
            continue
        rows.append(
            {
                name: line[begin:end].strip()
                for name, (begin, end) in zip(headers, spans, strict=True)
            }
        )
    return rows


def kv_value(stdout: str, label: str) -> str:
    """Read one ``cli._kv`` line — the ``  label       value`` pairs in a ``runs show`` header.

    Matched on the full padded label rather than by searching for the value, so an assertion
    about ``status`` cannot accidentally be satisfied by the word appearing inside ``error``
    or inside the JSON payload block further down the page.
    """
    match = re.search(rf"^  {re.escape(label).ljust(_LABEL_WIDTH)}(.*)$", stdout, re.MULTILINE)
    assert match is not None, f"no {label!r} line in the header:\n{stdout}"
    return match.group(1).strip()


#: One ``_configure_logging`` line: ``<iso8601>Z LEVEL   pid=N logger.name: message``.
#: Anchored on the ``pid=`` field so it cannot match a message that merely contains a colon.
_LOG_LINE = re.compile(r"^\S+Z\s+\w+\s+pid=\d+\s+[\w.]+: (?P<message>.*)$", re.MULTILINE)


def log_messages(stderr: str) -> list[str]:
    """Strip the log prefix off each stderr line, leaving the message the CLI wrote.

    Necessary for any assertion about how a message *starts*. Every diagnostic reaches
    stderr through ``logging``, so ``stderr`` itself always begins with a timestamp — which
    makes ``stderr.startswith(...)`` and ``stderr.lstrip().startswith(...)`` unfalsifiable,
    and an assertion that cannot fail is worse than no assertion at all. This is how a test
    asserts on the first character of the message body rather than on the first character of
    the log line.
    """
    return [match.group("message") for match in _LOG_LINE.finditer(stderr)]


def _int(cell: str) -> int:
    """Read a ``cli._count`` cell: thousands-separated, or ``-`` for a missing value."""
    return 0 if cell == "-" else int(cell.replace(",", ""))


def _money(cell: str) -> Decimal:
    """Read a ``cli._usd`` cell. ``-`` means "no budget", which is not zero."""
    return Decimal(cell)


# --------------------------------------------------------------------------------------
# row builders
# --------------------------------------------------------------------------------------


def insert_run(
    db: Database,
    *,
    run_id: str | None = None,
    pipeline_name: str = "cli_probe",
    tenant_id: str = TENANT,
    status: str = "queued",
    spent_usd: str = "0",
    attempt: int = 0,
    failures: int = 0,
    max_attempts: int = 5,
    error_class: str | None = None,
    locked_by: str | None = None,
    retry_in_seconds: float | None = None,
) -> UUID:
    """Insert a ``runs`` row in a precise state, bypassing the worker entirely.

    ``llmrun runs list`` and ``llmrun health`` are pure read commands: what they have to get
    right is how they *render* a row, and driving a real worker into each of the six statuses
    would test the worker instead. ``locked_at`` is stamped ``now()`` whenever ``locked_by``
    is given, because a lease with no timestamp is a state the worker cannot produce.
    """
    with transaction(db.conn) as cur:
        cur.execute(
            _INSERT_RUN,
            (
                run_id,
                pipeline_name,
                tenant_id,
                status,
                "{}",
                spent_usd,
                attempt,
                failures,
                max_attempts,
                error_class,
                locked_by,
                locked_by,
                retry_in_seconds,
            ),
        )
        row = cur.fetchone()
    assert row is not None
    return row["id"]


def insert_call(
    db: Database,
    run_id: UUID,
    *,
    step_name: str = "answer",
    input_hash: str = "hash-0",
    model: str = PRICED_MODEL,
    tokens_in: int = 100,
    tokens_out: int = 200,
    cost_usd: str = "0",
    cached: bool = False,
    age_days: float = 0.0,
) -> None:
    """Append a ledger row aged ``age_days`` into the past."""
    with transaction(db.conn) as cur:
        cur.execute(
            _INSERT_CALL,
            (
                str(run_id),
                step_name,
                0,
                input_hash,
                model,
                tokens_in,
                tokens_out,
                cost_usd,
                cached,
                age_days * 86_400.0,
            ),
        )


# ======================================================================================
#  enqueue
# ======================================================================================


def test_enqueue_prints_a_run_id_and_nothing_else(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """``run_id=$(llmrun enqueue ...)`` is documented usage, so stdout is a contract.

    The id has to be the *only* thing on stdout and it has to be parseable — the epilog in
    ``cli.py`` shows exactly that shell substitution, and ``_cmd_enqueue`` routes its "queued
    X for tenant Y" line through ``logger`` to stderr specifically to keep it that way. One
    ``print`` added for friendliness and every script capturing the id gets two lines and a
    ``ValueError`` from whatever consumes it next.

    Also pins that enqueue *only* enqueues: the row lands in ``queued`` with the arguments
    stored as jsonb, and no work has started.
    """
    result = run_cli("enqueue", "cli_probe", "--tenant", TENANT, "--json", '{"topic": "alpha"}')

    assert result.exit_code == 0, result.stderr
    assert len(result.lines) == 1, f"stdout was not a bare run id: {result.stdout!r}"
    run_id = UUID(result.lines[0].strip())

    record = db.run(run_id)
    assert record.status == "queued"
    assert record.input == {"topic": "alpha"}
    assert record.pipeline == "cli_probe"


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ("[1, 2]", "a JSON array is valid JSON and still not a keyword-argument mapping"),
        ('{"topic": ', "truncated by a shell that ate the closing brace"),
        ('{"topic": NaN}', "Python's extension to JSON, which jsonb cannot store"),
    ],
    ids=["not-an-object", "malformed", "nan"],
)
def test_enqueue_rejects_bad_json_as_usage_with_a_silent_stdout(
    db: Database,
    run_cli: Callable[..., CliResult],
    payload: str,
    why: str,
) -> None:
    """All three failures of ``--json`` are the operator's, so all three must exit 2.

    ``NaN`` is the one that motivated this test. ``json.loads`` accepts the bare tokens
    ``NaN``, ``Infinity`` and ``-Infinity`` by default — they are not in RFC 8259 and ``jsonb``
    will not store them — so that input sailed through the only check ``--json`` gets and died
    two layers down inside the INSERT, as a driver error with exit 1 and a leaked
    ``unnamed portal parameter`` line, while every *other* malformed ``--json`` was caught at
    the boundary and exited 2. Same class of mistake, two different exit codes, one of them
    telling the operator their database is broken.

    stdout is asserted empty on all three because a script doing ``run_id=$(llmrun enqueue
    ...)`` checks ``$?`` if it checks anything, and a partial or diagnostic line on stdout
    becomes a run id in somebody's variable.
    """
    result = run_cli("enqueue", "cli_probe", "--tenant", TENANT, "--json", payload)

    assert result.exit_code == 2, f"{why}: expected a usage error, got {result.exit_code}"
    assert result.stdout == "", f"{why}: bad input still wrote to stdout"
    with transaction(db.conn) as cur:
        cur.execute("SELECT count(*) AS n FROM runs")
        row = cur.fetchone()
    assert row is not None and row["n"] == 0, f"{why}: a run was enqueued anyway"


def test_enqueue_of_an_unregistered_pipeline_points_at_import(
    run_cli: Callable[..., CliResult],
) -> None:
    """An unknown pipeline name is bad usage, not a crash, and the fix goes in the message.

    ``registry.get`` raises ``KeyError``, whose ``str()`` is the message wrapped in quotes;
    ``_resolve_pipeline`` unwraps it and re-raises as a usage error so the operator sees the
    ``--import`` advice rather than a Python repr, and gets exit 2 rather than the exit 1 that
    would otherwise say "the command failed" about a typo.

    The unwrapping is asserted on the *message body*, not on ``stderr``. ``str(KeyError(msg))``
    is ``repr(msg)``, and because the registry's message contains single quotes around the
    pipeline name that repr is double-quoted — so dropping ``.strip("\\"'")`` puts a stray ``"``
    at the front of the line an operator reads. It is a one-character difference and the only
    way to see it is to look at the character, which means looking past the log prefix every
    line on stderr carries.
    """
    result = run_cli("enqueue", "not_a_pipeline", "--tenant", TENANT)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--import" in result.stderr

    first = log_messages(result.stderr)[0]
    assert first.startswith("no pipeline named"), (
        f"the KeyError repr leaked into the message the operator reads: {first!r}"
    )


# ======================================================================================
#  runs list
# ======================================================================================


def test_runs_list_refuses_a_limit_below_one(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """``--limit 0`` must be an argument error, not an empty report.

    Without the check the value goes straight into ``LIMIT %s``: Postgres accepts ``LIMIT 0``
    happily and returns nothing, so the command prints "no runs match" and exits 0 — telling
    the operator their queue is empty when it is not. That is the worst possible answer from a
    diagnostic command, and it is indistinguishable from the truth. (``LIMIT -1`` is a
    database error and would exit 1, which is the wrong code for a typo.)
    """
    insert_run(db)

    for limit in ("0", "-1"):
        result = run_cli("runs", "list", "--limit", limit)
        assert result.exit_code == 2, f"--limit {limit} was accepted"
        assert result.stdout == "", f"--limit {limit} printed a report"


def test_runs_list_filters_by_tenant_and_by_status(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """The two filters are what make the command usable on a shared queue.

    A multi-tenant deployment's ``runs`` table is one table; ``--tenant`` is the only thing
    separating one customer's runs from another's on screen, and a filter that silently
    matched everything would show an operator rows they should not be looking at. ``--status``
    is how "what is stuck?" is asked. Both are parameterised into the same statement with
    ``(%s::text IS NULL OR col = %s::text)``, so a mistake in either half of that pattern
    turns the filter into a no-op that still looks like it worked.
    """
    mine = insert_run(db, tenant_id="acme", status="queued")
    theirs = insert_run(db, tenant_id="globex", status="queued")
    finished = insert_run(db, tenant_id="acme", status="succeeded")

    by_tenant = run_cli("runs", "list", "--tenant", "acme")
    assert by_tenant.exit_code == 0, by_tenant.stderr
    listed = {row["RUN ID"] for row in parse_table(by_tenant.lines)}
    assert listed == {str(mine), str(finished)}
    assert str(theirs) not in by_tenant.stdout, "another tenant's run appeared in the report"

    by_status = run_cli("runs", "list", "--status", "succeeded")
    assert by_status.exit_code == 0, by_status.stderr
    assert {row["RUN ID"] for row in parse_table(by_status.lines)} == {str(finished)}

    both = run_cli("runs", "list", "--tenant", "globex", "--status", "succeeded")
    assert both.exit_code == 0, both.stderr
    assert both.stdout.strip() == "no runs match"

    bogus = run_cli("runs", "list", "--status", "nonsense")
    assert bogus.exit_code == 2, "--status accepted a value runs.status cannot hold"
    assert bogus.stdout == ""


def test_runs_list_lease_column_separates_held_from_backing_off_from_finished(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """LEASE is the column that answers "is anything actually happening to this run?".

    ``status`` cannot: a ``running`` row means either "a worker is on it right now" or "a
    worker died holding it and the lease has not expired yet", and those need opposite
    responses. Neither can ``attempt``. So the three states have to be visually distinct in
    the list itself, or triage becomes a ``runs show`` on every candidate:

    * held — the worker id, so the next question ("which box?") is already answered;
    * backing off — how long until it is claimable again, which is why a queued run can sit
      there doing nothing and be perfectly healthy;
    * terminal — blank, because a finished run has no lease and saying "none" on every row of
      a mostly-finished table is noise.

    Delete this and the column can quietly become decorative — ``_lease_summary`` returning
    "" for everything still renders a perfectly well-formed table.
    """
    held = insert_run(db, status="running", locked_by="box-a/4242", attempt=1)
    backing_off = insert_run(db, status="queued", attempt=1, failures=1, retry_in_seconds=300.0)
    finished = insert_run(db, status="succeeded", attempt=1)

    result = run_cli("runs", "list")
    assert result.exit_code == 0, result.stderr

    rows = {row["RUN ID"]: row for row in parse_table(result.lines)}
    assert set(rows) == {str(held), str(backing_off), str(finished)}

    assert rows[str(held)]["LEASE"].startswith("box-a/4242"), (
        "a run being worked on right now does not say who is working on it"
    )
    assert rows[str(backing_off)]["LEASE"].startswith("retry "), (
        "a run waiting out its run-level backoff is indistinguishable from an idle queued run"
    )
    assert rows[str(finished)]["LEASE"] == "", "a terminal run reported a lease it cannot hold"

    # ATT is `attempt/max_attempts`, not a bare number: "3" means nothing without the
    # ceiling it is being measured against, and the ceiling is per-run (copied from the
    # pipeline at enqueue time), so it cannot be inferred from a global default.
    assert rows[str(held)]["ATT"] == "1/5"


# ======================================================================================
#  runs show
# ======================================================================================


def test_runs_show_marks_replayed_calls_and_prices_what_the_replay_saved(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """The money shot: the command that turns "we do not double-bill" into something readable.

    The whole path is driven through the CLI — enqueue, worker, worker again — because the
    claim is about what an operator can *see*, and the engine's own correctness is already
    covered by ``tests/test_worker.py`` and the chaos suite. What is reproduced here is the
    crash window the ``llm_calls`` ledger exists for: the provider call committed, the step
    checkpoint not. Deleting the checkpoints of a finished run puts it exactly there, and it
    is the only way to make a replay observable — with checkpoints intact a resumed run
    replays from the ``steps`` table and never consults the ledger, so ``cached`` rows would
    never be written and there would be nothing for this command to mark.

    Three independent signals are asserted, because this table is the evidence for the
    project's central claim and evidence that needs explaining is not evidence: the ``*``
    gutter, the ``CACHED`` billing column, and a non-zero SAVED USD. The last one is the only
    one that requires the arithmetic in ``_replay_savings`` to be right — a replayed row
    stores ``cost_usd = 0`` by design, so the value of the hit lives on the *billed* row
    sharing its ``input_hash`` and has to be joined back to it. Get that wrong and the report
    says a crash saved you nothing, which is the opposite of the claim being made.
    """
    enqueued = run_cli("enqueue", "cli_probe", "--tenant", TENANT, "--json", '{"topic": "alpha"}')
    run_id = UUID(enqueued.lines[0].strip())

    first = run_cli("worker", "--once", "--provider", "echo", "--model", PRICED_MODEL)
    assert first.exit_code == 0, first.stderr
    assert first.stdout.strip() == str(run_id)
    assert db.run(run_id).status == "succeeded"
    assert db.count_llm_calls(run_id, cached=False) == 1

    with transaction(db.conn) as cur:
        cur.execute(_DELETE_STEPS, (str(run_id),))
    db.requeue(run_id)
    db.clear_backoff(run_id)

    second = run_cli("worker", "--once", "--provider", "echo", "--model", PRICED_MODEL)
    assert second.exit_code == 0, second.stderr
    assert db.run(run_id).status == "succeeded"
    assert db.count_llm_calls(run_id, cached=False) == 1, "the resumed run re-bought a completion"
    assert db.count_llm_calls(run_id, cached=True) == 1, "the absorbed call left no evidence"

    shown = run_cli("runs", "show", str(run_id))
    assert shown.exit_code == 0, shown.stderr

    ledger = parse_table(shown.lines, after="LLM CALLS")
    assert len(ledger) == 2
    billed, replayed = ledger

    assert billed[""] == "" and billed["BILLING"] == "billed"
    assert replayed[""] == "*", "the replayed call has no gutter mark"
    assert replayed["BILLING"] == "CACHED"

    paid = _money(billed["COST USD"])
    assert paid > 0, f"{PRICED_MODEL} priced a real call at zero; the saving cannot be shown"
    assert _money(replayed["COST USD"]) == 0, "a replay was billed"
    assert _money(replayed["SAVED USD"]) == paid, (
        "the replayed call is not priced against the billed call that shares its input_hash, "
        "so the report claims the crash saved nothing"
    )

    assert kv_value(shown.stdout, "status") == "SUCCEEDED"
    assert kv_value(shown.stdout, "cache") == f"1 of 2 calls replayed, ${paid:.6f} not spent again"
    assert kv_value(shown.stdout, "attempts") == "2 of 3 allowed", (
        "the header does not report the run's own ceiling, so a run one claim from being "
        "unclaimable looks identical to a fresh one"
    )
    assert f"an unguarded retry would have spent ${paid:.6f}" in shown.stdout


def test_runs_show_refuses_an_ambiguous_prefix_and_names_the_candidates(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """A truncated id that matches two runs must not resolve to whichever sorts first.

    ``runs show`` takes a prefix because that is what a terminal, a chat message or a log line
    leaves you with. The cost of that convenience is collisions, and picking one silently
    would print a full, confident, entirely wrong report — the header, the steps and the cost
    ledger of a different run — with nothing on screen hinting that a choice was made. So the
    ambiguity is an error, and the error lists the candidates, because the operator's next
    command is ``runs show`` with one of them and they should not have to go and find them.

    The ids are pinned rather than generated: ``gen_random_uuid`` will not hand out two values
    sharing an eight-character prefix inside the lifetime of this project.
    """
    first = insert_run(db, run_id="abcd1234-0000-4000-8000-000000000001")
    second = insert_run(db, run_id="abcd1234-0000-4000-8000-000000000002")
    other = insert_run(db, run_id="99990000-0000-4000-8000-000000000003")

    ambiguous = run_cli("runs", "show", "abcd1234")
    assert ambiguous.exit_code == 1, "an ambiguous prefix resolved to a single run"
    assert ambiguous.stdout == "", "a report was printed for an id that was not resolved"
    assert "ambiguous" in ambiguous.stderr
    assert str(first) in ambiguous.stderr and str(second) in ambiguous.stderr, (
        "the operator is told their prefix is ambiguous but not what it is ambiguous between"
    )

    # ...and a prefix long enough to be unique still works, which is the point of accepting
    # prefixes at all. Without this the test above would also pass if prefixes were rejected.
    unique = run_cli("runs", "show", "9999")
    assert unique.exit_code == 0, unique.stderr
    assert f"RUN {other}" in unique.stdout


def test_runs_show_reports_an_unresolvable_id_without_a_traceback(
    run_cli: Callable[..., CliResult],
) -> None:
    """The two ways an id can fail to resolve are different mistakes and get different codes.

    A well-formed id that matches nothing is a *command* failure (exit 1): the argument was
    fine, the run is not there. A token that is not an id at all — a pasted word, a shell
    glob, a truncated log line — is a *usage* failure (exit 2), and catching it before the
    query is what stops ``runs show "'; DROP"`` reaching the database as a LIKE pattern.
    Collapsing the two would make a script unable to tell "this run was pruned" from "your
    argument is nonsense".
    """
    missing = run_cli("runs", "show", "00000000-0000-4000-8000-000000000000")
    assert missing.exit_code == 1
    assert missing.stdout == ""
    assert "no run matches" in missing.stderr

    nonsense = run_cli("runs", "show", "not-a-run-id")
    assert nonsense.exit_code == 2
    assert nonsense.stdout == ""
    assert "neither a run id nor a run id prefix" in nonsense.stderr


# ======================================================================================
#  costs
# ======================================================================================


def test_costs_totals_equal_the_sum_of_the_ledger(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """The report is an invoice reconciliation, so its arithmetic is the whole product.

    Four properties in one fixture, because they are four ways the same query can be subtly
    wrong and each one is silent:

    * SPEND is the sum of every ledger row, per group and in total. GROUPING SETS appends the
      total as one more row rather than a second query, so a mistake in the ``grouping()``
      flag prints a group as the total or drops the total entirely.
    * RUNS is ``count(DISTINCT run_id)`` at *both* levels. A run spans several models, so
      adding the per-group run counts up would over-count it — which is why the total cannot
      simply be a sum of the rows above it.
    * TOK IN / TOK OUT count only *billed* calls. A replayed row deliberately copies the
      original's token counts so the saving can be priced; adding those into the totals would
      report tokens nobody bought, sitting next to a spend figure that correctly excludes
      them, and the two numbers would stop dividing into a plausible rate.
    * SAVED is the cost the replays would have re-incurred, recovered by joining each cached
      row back to the billed row sharing its ``input_hash``. A cached row stores zero, so
      without the join the saving is invisible and the report's headline is always $0.
    """
    run_id = insert_run(db, status="succeeded")
    # (input_hash, model, tokens_in, tokens_out, cost, cached). The last row is a replay of
    # the first: free by construction, carrying the original's token counts — which is what
    # makes it the trap for both the TOK IN filter and the SAVED USD join.
    ledger_rows = (
        ("h1", PRICED_MODEL, 100, 200, "0.002000", False),
        ("h2", PRICED_MODEL, 150, 250, "0.003000", False),
        ("h3", "claude-opus-5", 400, 800, "0.005000", False),
        ("h1", PRICED_MODEL, 100, 200, "0", True),
    )
    for digest, model, tokens_in, tokens_out, cost, cached in ledger_rows:
        insert_call(
            db,
            run_id,
            input_hash=digest,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            cached=cached,
        )

    result = run_cli("costs")
    assert result.exit_code == 0, result.stderr

    rows = {row["MODEL"]: row for row in parse_table(result.lines)}
    assert set(rows) == {PRICED_MODEL, "claude-opus-5", ""}
    total = rows[""]
    assert total["PIPELINE"] == "TOTAL"

    haiku = rows[PRICED_MODEL]
    assert _int(haiku["BILLED"]) == 2
    assert _int(haiku["REPLAYED"]) == 1
    assert _int(haiku["TOK IN"]) == 250, "a replayed call's tokens were counted as purchased"
    assert _int(haiku["TOK OUT"]) == 450
    assert _money(haiku["SPEND USD"]) == Decimal("0.005000")
    assert _money(haiku["SAVED USD"]) == Decimal("0.002000")

    assert _money(total["SPEND USD"]) == Decimal("0.010000")
    assert _money(total["SAVED USD"]) == Decimal("0.002000")
    assert _int(total["BILLED"]) == 3
    assert _int(total["REPLAYED"]) == 1
    assert _int(total["RUNS"]) == 1, "one run spanning two models was counted twice"
    assert "1 replayed call(s) cost nothing: $0.002000" in result.stdout

    # And the scoping: another tenant's spend must not land in this tenant's invoice.
    other = insert_run(db, tenant_id="somebody-else", status="succeeded")
    insert_call(db, other, input_hash="h9", cost_usd="9.000000")
    scoped = run_cli("costs", "--tenant", TENANT)
    assert scoped.exit_code == 0, scoped.stderr
    scoped_total = next(row for row in parse_table(scoped.lines) if row["PIPELINE"] == "TOTAL")
    assert _money(scoped_total["SPEND USD"]) == Decimal("0.010000")


def test_costs_defaults_to_the_last_thirty_days(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """An unbounded aggregate over an append-only ledger is a query that gets slower forever.

    Nothing prunes ``llm_calls``, so "all time" is a report whose cost grows with every call
    the system has ever made and whose answer is almost never the one anybody wanted. The
    default window is therefore 30 days — which makes the *stated window* part of the
    contract: a total silently covering a month reads exactly like a total covering all time,
    and the difference is the whole invoice. Hence the ``logger.info`` line naming the bound.

    Delete this and the default can revert to unbounded without a single other test noticing,
    because every other cost assertion inserts its rows at ``now()``.
    """
    run_id = insert_run(db, status="succeeded")
    insert_call(db, run_id, input_hash="recent", cost_usd="0.001000", age_days=1.0)
    insert_call(db, run_id, input_hash="ancient", cost_usd="7.000000", age_days=45.0)

    result = run_cli("costs")
    assert result.exit_code == 0, result.stderr

    total = next(row for row in parse_table(result.lines) if row["PIPELINE"] == "TOTAL")
    assert _money(total["SPEND USD"]) == Decimal("0.001000"), (
        "a 45-day-old ledger row was included in the default report, so the window is not "
        "being applied and the query scans the whole lifetime ledger"
    )
    assert _int(total["BILLED"]) == 1
    assert "since " in result.stderr, "the report does not state the window it covers"

    # The old row is not lost, merely out of view: asking for it explicitly finds it.
    widened = run_cli("costs", "--since", "60")
    widened_total = next(row for row in parse_table(widened.lines) if row["PIPELINE"] == "TOTAL")
    assert _money(widened_total["SPEND USD"]) == Decimal("7.001000")


def test_costs_since_and_until_bound_the_window_at_both_ends(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """Both bounds, and the half-open convention that makes adjacent windows sum correctly.

    ``--since`` is inclusive and ``--until`` is exclusive, so ``[Jan, Feb)`` and ``[Feb, Mar)``
    partition the ledger instead of double-counting whatever landed exactly on the boundary.
    A report used for chargeback that counts a row in two months is worse than one that counts
    it in neither, because it reconciles against nothing and nobody notices until a customer
    adds the invoices up.
    """
    run_id = insert_run(db, status="succeeded")
    insert_call(db, run_id, input_hash="old", cost_usd="1.000000", age_days=20.0)
    insert_call(db, run_id, input_hash="mid", cost_usd="0.100000", age_days=10.0)
    insert_call(db, run_id, input_hash="new", cost_usd="0.010000", age_days=1.0)

    windowed = run_cli("costs", "--since", "15", "--until", "5")
    assert windowed.exit_code == 0, windowed.stderr
    total = next(row for row in parse_table(windowed.lines) if row["PIPELINE"] == "TOTAL")
    assert _money(total["SPEND USD"]) == Decimal("0.100000"), (
        "--since/--until did not exclude the rows on either side of the window"
    )
    assert _int(total["BILLED"]) == 1
    assert "until " in windowed.stderr, "the report does not state its upper bound"

    # An empty window is a legitimate answer, not a failure: exit 0, and say so plainly.
    empty = run_cli("costs", "--since", "4", "--until", "2")
    assert empty.exit_code == 0, empty.stderr
    assert "no llm calls recorded yet" in empty.stdout


def test_costs_refuses_a_window_that_ends_before_it_begins(
    run_cli: Callable[..., CliResult],
) -> None:
    """``--until`` before ``--since`` can only ever match nothing, so say so instead.

    Passed through, the query returns an empty result and the command prints "no llm calls
    recorded yet" and exits 0 — which is a true statement about an impossible window and reads
    exactly like "this tenant has spent nothing". Somebody transposing two dates deserves to
    be told, not reassured. An unparseable bound is the same class of mistake and gets the
    same code.
    """
    inverted = run_cli("costs", "--since", "2026-01-01", "--until", "2025-01-01")
    assert inverted.exit_code == 2
    assert inverted.stdout == ""
    assert "must be after" in inverted.stderr

    unparseable = run_cli("costs", "--since", "last tuesday")
    assert unparseable.exit_code == 2
    assert unparseable.stdout == ""
    assert "ISO-8601" in unparseable.stderr


# ======================================================================================
#  health
# ======================================================================================


def test_health_reports_the_backlog_and_counts_a_run_at_its_ceiling_as_stranded(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """The command a readiness probe calls, and the one number in it nothing else surfaces.

    ``queued`` and ``running`` are backlog. ``stranded`` is different in kind: a run that has
    used every allowed execution and has not yet been given a terminal status by the worker's
    sweep. Those rows are invisible everywhere else — they read as ordinary work in progress
    in ``runs list``, the claim query will not take them (it gates on ``attempt <
    max_attempts``), and no error has been recorded because in the usual case nothing survived
    to record one. A growing count means runs are burning executions without producing
    verdicts, which is a symptom rather than a state, hence the WARNING alongside the number.

    Drop the ``attempt >= max_attempts`` filter and ``stranded`` becomes a second copy of
    ``queued + running`` — a metric that is never zero and therefore never means anything.
    """
    insert_run(db, status="queued", attempt=0, max_attempts=5)
    insert_run(db, status="running", attempt=1, max_attempts=5, locked_by="box-a/1")
    at_ceiling = insert_run(db, status="queued", attempt=5, max_attempts=5)

    result = run_cli("health")
    assert result.exit_code == 0, result.stderr

    counts = dict(re.findall(r"(\w+)=(\d+)", result.stdout))
    assert result.stdout.startswith("ok  ")
    assert counts["queued"] == "2", "the run at its ceiling is still queued and must be counted"
    assert counts["running"] == "1"
    assert counts["stranded"] == "1", (
        f"run {at_ceiling} has used all 5 of its allowed executions and is unclaimable, but "
        "health did not count it as stranded"
    )

    with transaction(db.conn) as cur:
        cur.execute(_COUNT_MIGRATIONS)
        row = cur.fetchone()
    assert row is not None
    assert counts["migrations"] == str(row["n"]), (
        "the schema-presence check reports a migration count that does not match the table"
    )
    assert "awaiting the sweep" in result.stderr, "a stranded run was counted but not flagged"


def test_health_stays_silent_about_the_sweep_when_nothing_is_stranded(
    db: Database,
    run_cli: Callable[..., CliResult],
) -> None:
    """The warning has to be conditional, or it is not a warning.

    A probe that logs "runs are awaiting the sweep" on every single call trains whoever reads
    the logs to ignore the line, and the one time it matters they will. This is the other half
    of the test above and the reason the count is checked rather than the message alone.
    """
    insert_run(db, status="queued", attempt=0, max_attempts=5)

    result = run_cli("health")
    assert result.exit_code == 0, result.stderr
    assert "stranded=0" in result.stdout
    assert "awaiting the sweep" not in result.stderr


# ======================================================================================
#  worker flags
# ======================================================================================


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lease-seconds", "0"),
        ("--lease-seconds", "-5"),
        ("--poll-interval", "0"),
        ("--poll-interval", "-1"),
    ],
)
def test_worker_refuses_timings_that_break_the_loop_rather_than_slow_it(
    run_cli: Callable[..., CliResult],
    flag: str,
    value: str,
) -> None:
    """Two flags with a value that silently destroys the system instead of degrading it.

    ``--lease-seconds 0`` collapses the lease window to nothing: every ``running`` row
    satisfies ``locked_at < now() - interval '0'`` the instant it is stamped, so every poll of
    every worker steals the run another worker is currently executing, and a pool burns each
    run's execution ceiling in seconds having completed no work. ``--poll-interval 0`` turns
    the idle loop into a busy loop issuing claim queries as fast as the socket allows.

    Neither announces itself. The worker logs ``lease=0s poll=0.0s`` once at startup and then
    looks perfectly healthy — same process, same log volume, no errors — which is why the
    refusal has to happen at the argument boundary, before anything is imported or connected.

    ``--once`` is passed so that a regression here fails the test in milliseconds instead of
    entering ``run_forever`` and hanging the suite.
    """
    result = run_cli("worker", "--once", flag, value)

    assert result.exit_code == 2, f"{flag} {value} was accepted"
    assert result.stdout == ""


def test_worker_rejects_an_unknown_provider_at_the_argument_boundary(
    run_cli: Callable[..., CliResult],
) -> None:
    """``--provider`` is constrained by argparse ``choices``, generated from PROVIDER_NAMES.

    Both spellings of this failure exit 2 — ``_cmd_worker`` catches ``get_provider``'s
    ``ValueError`` for the ``$LLMRUN_PROVIDER`` path argparse never sees — so the exit code
    alone proves nothing, and the assertion is on argparse's own "invalid choice" message.
    That is the difference between rejecting a typo before importing modules and opening a
    database connection, and rejecting it after. It also makes ``--help`` and shell completion
    correct for free, and pins the list to ``providers.base.PROVIDER_NAMES`` so the flag and
    the factory cannot drift apart.
    """
    result = run_cli("worker", "--once", "--provider", "bogus")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "invalid choice" in result.stderr, (
        "the provider name was not rejected by argparse, so a typo reaches get_provider only "
        "after --import has run and a connection has been opened"
    )
    for known in ("echo", "anthropic"):
        assert known in result.stderr, "the rejection does not list the providers that do exist"
