"""The claim loop's two counters, the lease, and what happens when workers overlap.

Everything here is about the gap between "a lease expired" and "the first worker is dead".
The lease cannot tell those apart — it only knows a worker has been quiet — so the claim
query, the retry budget, the lease fence and the billed call each have to behave correctly
when the guess is wrong. The properties, one per section:

* a run whose worker dies without writing a verdict cannot be re-claimed forever;
* lease churn must not spend a retry budget that exists for genuine failures;
* a pool of workers polling one queue must hand every run to exactly one of them;
* two workers executing the same run at the same time must not both pay for the same call,
  must not overwrite each other's checkpoints, and must not steal a run that is visibly
  making progress;
* a worker that cannot serve the queue it is polling must hand runs back rather than fail
  them, and a worker configured with a lease or poll interval that breaks the queue must
  refuse to start at all.

The concurrency sections need real concurrency — several connections, several threads, one
Postgres row lock — because the bugs they guard against live entirely in the ordering of
transactions. None of them are built out of wall-clock sleeps: every interleaving below is
forced with :class:`threading.Event` / :class:`threading.Barrier`, so the test either proves
the property or hangs, rather than passing on a fast machine and failing on a loaded one.
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from llmrun.context import RunContext
from llmrun.db import connect, transaction
from llmrun.errors import PermanentError, TransientError
from llmrun.pipeline import pipeline
from llmrun.providers.echo import EchoProvider
from llmrun.worker import run_once, validate_worker_timings

if TYPE_CHECKING:
    from uuid import UUID

    import psycopg
    from psycopg.rows import DictRow

    from conftest import Database
    from llmrun.types import LLMResult

WORKER_ID = "test-worker"

#: Identities for the two-worker tests. Distinct strings on purpose: the lease fence is a
#: string comparison against ``runs.locked_by``, so two contexts sharing one worker id would
#: pass every fence and prove nothing.
HOLDER_ID = "holder-worker"
THIEF_ID = "thief-worker"

#: A pipeline name no ``@pipeline`` in this process registers, i.e. exactly what a worker
#: started with the wrong ``--import`` sees for every run in a shared queue.
UNREGISTERED_PIPELINE = "test_worker_pipeline_this_process_never_imported"

#: Run inputs whose step body should raise a :class:`TransientError` every time. Keyed on
#: the *argument* rather than the step name because that is what the step body is handed.
ALWAYS_FLAKY: set[str] = set()

_EXPIRE_LEASE = "UPDATE runs SET locked_at = now() - make_interval(secs => %s) WHERE id = %s::uuid"
_STEAL_LEASE = "UPDATE runs SET locked_by = %s WHERE id = %s::uuid"
_TAKE_LEASE = """
    UPDATE runs SET status='running', locked_by=%s, locked_at=now() WHERE id=%s::uuid
"""
_READ_COUNTERS = "SELECT attempt, failures, status FROM runs WHERE id = %s::uuid"
_READ_LEASE = "SELECT locked_by, locked_at FROM runs WHERE id = %s::uuid"


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    ALWAYS_FLAKY.clear()


def flaky(name: str) -> dict[str, Any]:
    if name in ALWAYS_FLAKY:
        raise TransientError(f"{name}: upstream is down")
    return {"name": name}


@pipeline(name="test_worker_flaky", budget_usd=None, max_attempts=3)
def flaky_pipeline(ctx: RunContext, name: str) -> dict[str, Any]:
    return ctx.step("work", flaky, name)


@pipeline(name="test_worker_slow", budget_usd=None, max_attempts=5)
def slow_pipeline(ctx: RunContext, name: str) -> dict[str, Any]:
    """One billed call, and nothing else — the smallest thing two workers can race over."""
    return {"answer": ctx.llm("answer", f"[{name}] answer the payload", {"name": name})}


def checkpointable(value: str) -> dict[str, Any]:
    """A step body with no failure modes of its own.

    The lease tests are about *who is allowed to write the checkpoint*, so the work itself
    has to be uninteresting: anything that could fail on its own would make a red test
    ambiguous between "the fence fired" and "the step broke".
    """
    return {"value": value}


class GatedEchoProvider(EchoProvider):
    """An echo provider that can be held still inside :meth:`complete`.

    This is what replaces the wall-clock sleeps the overlapping-worker test used to
    synchronise on. A ``latency_seconds`` provider makes the interleaving *likely*; two
    events make it *certain*, and turn "the second worker probably got there in time" into
    "the second worker is provably inside the window".

    ``entered`` is set on the way in, so the main thread knows the caller is now inside the
    provider call — holding the run's row lock, with its ledger row still uncommitted.
    ``release`` is waited on before answering, so the caller stays there until the test says
    otherwise. The wait is bounded: a test that deadlocks should fail with a message rather
    than hang until pytest is killed.
    """

    def __init__(self, *, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def complete(self, **params: Any) -> LLMResult:
        """Announce arrival, wait for permission, then answer exactly as ``EchoProvider``."""
        self.entered.set()
        if not self.release.wait(timeout=30):
            raise AssertionError("the gated provider was never released; the test deadlocked")
        return super().complete(**params)


class ObservingContext(RunContext):
    """A context that records what its *free* ledger lookup saw, and says when it happened.

    The overlapping-worker property is precisely "the free lookup missed, and the run was
    saved by the second lookup taken behind the row lock". A test that cannot tell whether
    the free lookup missed cannot tell those two lookups apart — it would pass just as
    happily against a build with no in-lock lookup at all, as long as the timing happened to
    let the free one hit. So the free lookup is observed directly rather than inferred.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.free_lookup_done = threading.Event()
        self.free_lookup_result: str | None = None

    def _replay_recorded_call(self, *, name: str, attempt: int, digest: str) -> str | None:
        text = super()._replay_recorded_call(name=name, attempt=attempt, digest=digest)
        self.free_lookup_result = text
        self.free_lookup_done.set()
        return text


def _claim(db: Database, *, provider: EchoProvider | None = None) -> Any:
    """One turn of a worker: sweep, claim, execute, record. Returns the run id or None."""
    return run_once(
        db.conn,
        worker_id=WORKER_ID,
        provider=provider or EchoProvider(),
        lease_seconds=30,
    )


def _expire_lease(db: Database, run_id: Any, *, seconds: int = 3600) -> None:
    """Backdate the lease so the next claim treats the run as abandoned."""
    with transaction(db.conn) as cur:
        cur.execute(_EXPIRE_LEASE, (float(seconds), str(run_id)))


def _counters(db: Database, run_id: Any) -> tuple[int, int, str]:
    with transaction(db.conn) as cur:
        cur.execute(_READ_COUNTERS, (str(run_id),))
        row = cur.fetchone()
    assert row is not None
    return int(row["attempt"]), int(row["failures"]), str(row["status"])


def _lease(db: Database, run_id: Any) -> tuple[str | None, datetime | None]:
    """``(locked_by, locked_at)`` — the two columns that *are* the lease.

    Read with SQL rather than through :class:`~llmrun.types.RunRecord`, which deliberately
    drops both: they are the worker's bookkeeping, not part of a run's public shape.
    """
    with transaction(db.conn) as cur:
        cur.execute(_READ_LEASE, (str(run_id),))
        row = cur.fetchone()
    assert row is not None
    return row["locked_by"], row["locked_at"]


def _take_lease(db: Database, run_id: Any, worker_id: str) -> None:
    """Put a run into ``running`` under ``worker_id``, as a fresh claim would leave it."""
    with transaction(db.conn) as cur:
        cur.execute(_TAKE_LEASE, (worker_id, str(run_id)))


# --------------------------------------------------------------------------------------
# the execution ceiling
# --------------------------------------------------------------------------------------


def test_a_run_that_kills_its_worker_is_not_re_claimed_forever(db: Database) -> None:
    """``max_attempts`` bounds executions, not just the failures a worker lived to record.

    A SIGKILLed worker never reaches the code that writes a verdict, so a run that reliably
    kills its worker used to be re-claimed on every lease expiry — burning a worker slot,
    and re-buying whatever it had already paid for on every cycle. Incrementing a counter
    nobody read bounded nothing; the ceiling has to live in the claim query itself.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    # Each iteration is a worker that claimed the run and then vanished without a verdict.
    claims = 0
    for _ in range(10):
        _expire_lease(db, run_id)
        if _claim(db) is None:
            break
        claims += 1
        # Undo the verdict the (surviving) worker wrote, so the run looks abandoned again.
        with transaction(db.conn) as cur:
            cur.execute(
                "UPDATE runs SET status='running', failures=0, locked_by=%s WHERE id=%s::uuid",
                (WORKER_ID, str(run_id)),
            )

    assert claims == 3, f"max_attempts=3 should allow exactly 3 executions, got {claims}"

    attempt, _, _ = _counters(db, run_id)
    assert attempt == 3


def test_a_run_stranded_past_its_ceiling_is_failed_rather_than_left_running(
    db: Database,
) -> None:
    """The companion sweep. A run nothing can claim must not read as work in progress.

    Without it the ceiling above trades an infinite loop for an invisible one: the row sits
    in ``running`` forever, indistinguishable in ``llmrun runs list`` from a run a worker is
    still busy with, and no status ever explains why nothing is happening.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    with transaction(db.conn) as cur:
        cur.execute(
            "UPDATE runs SET status='running', attempt=3, locked_by=%s WHERE id=%s::uuid",
            (WORKER_ID, str(run_id)),
        )
    _expire_lease(db, run_id)

    assert _claim(db) is None

    record = db.run(run_id)
    assert record.status == "failed"
    assert record.error_class == "permanent"
    assert "without recording a verdict" in (record.error_detail or "")


def test_a_run_still_inside_its_lease_is_not_swept(db: Database) -> None:
    """The sweep must not fail a run out from under a worker that is still working on it.

    A run on its final allowed attempt is *supposed* to be sitting in ``running`` with the
    ceiling reached. Only silence longer than the lease distinguishes that from a corpse.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    with transaction(db.conn) as cur:
        cur.execute(
            "UPDATE runs SET status='running', attempt=3, locked_by=%s, locked_at=now() "
            "WHERE id=%s::uuid",
            (WORKER_ID, str(run_id)),
        )

    assert _claim(db) is None
    assert db.run(run_id).status == "running", "a live worker's run was swept"


# --------------------------------------------------------------------------------------
# the retry budget
# --------------------------------------------------------------------------------------


def test_lease_churn_does_not_consume_the_retry_budget(db: Database) -> None:
    """A slow pipeline must still get its retries.

    ``attempt`` counts claims, including lease-expiry steals; the retry budget is about
    failures. Measuring one with the other means a pipeline whose steps legitimately outlive
    ``--lease-seconds`` reaches its first genuine transient error with the budget already
    spent, and is failed terminally having never been retried once — a ``failed`` run with
    ``error_class='transient'`` that nobody ever gave a second chance.
    """
    ALWAYS_FLAKY.add("alpha")
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    # One claim lost to lease expiry, no failure recorded by it. max_attempts is 3, so the
    # run still has executions in hand — which is the case this test is about. What happens
    # when churn eats the *last* execution is a different property, tested below.
    with transaction(db.conn) as cur:
        cur.execute(
            "UPDATE runs SET status='running', attempt=attempt+1, locked_by=%s WHERE id=%s::uuid",
            (WORKER_ID, str(run_id)),
        )
    _expire_lease(db, run_id)

    attempt, failures, _ = _counters(db, run_id)
    assert (attempt, failures) == (1, 0)

    # Now the first genuine failure. It must be retried, not treated as terminal.
    assert _claim(db)
    record = db.run(run_id)
    assert record.status == "queued", (
        "the first real failure was treated as terminal because lease churn had already "
        "spent the retry budget"
    )
    assert record.failures == 1

    # ...and "queued" has to mean claimable. A requeued run that the claim query will not
    # take back is not on the queue, it is stranded — see the regression test below.
    db.clear_backoff(run_id)
    assert _claim(db) is not None, "the requeued run was not claimable"


def test_a_run_requeued_at_its_execution_ceiling_is_never_stranded(db: Database) -> None:
    """A requeue must never produce a row the claim query will not take back.

    The two counters bound different things: ``failures`` says whether the run deserves
    another try, ``attempt`` says whether it can be given one. Deciding the requeue from
    ``failures`` alone wrote ``status='queued' AND attempt >= max_attempts`` — rejected by
    ``_CLAIM_SQL`` (which gates on ``attempt < max_attempts``) and invisible to the sweep
    (which only looked at ``running``). The run then sat in ``queued`` forever: no error, no
    terminal status, no work being done, and nothing in the system able to notice.

    Lease churn reaches that state without a single genuine failure, because every steal
    burns an ``attempt`` and no ``failures`` — which is exactly the setup below.
    """
    ALWAYS_FLAKY.add("alpha")
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    # Churn away every execution but one, recording no failures at all.
    for _ in range(2):
        with transaction(db.conn) as cur:
            cur.execute(
                "UPDATE runs SET status='running', attempt=attempt+1, locked_by=%s "
                "WHERE id=%s::uuid",
                (WORKER_ID, str(run_id)),
            )
        _expire_lease(db, run_id)

    assert _counters(db, run_id) == (2, 0, "running")

    # The final allowed execution, and it fails. `failures` (1) is still under the ceiling
    # (3), so the old code requeued — into a state nothing could claim.
    assert _claim(db)

    record = db.run(run_id)
    assert record.attempt == 3, "this was the last allowed execution"
    assert record.status != "queued", (
        "the run was requeued after using every allowed execution: the claim query gates on "
        "attempt < max_attempts, so nothing will ever pick this row up again"
    )
    assert record.status == "failed"
    assert "runs.attempt" in (record.error_detail or ""), (
        "a run failed with retries still unspent must say why, or it reads as the retry "
        "budget being ignored"
    )


def test_the_sweep_rescues_a_run_already_stranded_in_queued(db: Database) -> None:
    """Rows written by the old code still have to reach a terminal status.

    Fixing the requeue stops new strandings; it does nothing for the rows already sitting in
    a database. The sweep covers ``queued`` as well as ``running`` for exactly that reason —
    and needs no lease check on this branch, because a queued row is held by nobody.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    with transaction(db.conn) as cur:
        cur.execute(
            "UPDATE runs SET status='queued', attempt=3, failures=1, locked_by=NULL, "
            "locked_at=NULL WHERE id=%s::uuid",
            (str(run_id),),
        )
    db.conn.commit()

    assert _claim(db) is None, "precondition: the stranded row is not claimable"

    record = db.run(run_id)
    assert record.status == "failed", "the sweep left a stranded run reading as queued work"
    assert "requeued after it had already used" in (record.error_detail or "")


def test_a_requeued_run_is_paced_rather_than_re_claimed_immediately(db: Database) -> None:
    """Run-level retries get the same jittered backoff the in-step loop has always had.

    Without it a requeued run was claimable on the very next poll, almost always by the
    worker that had just failed it — so a provider having a bad ten seconds was answered
    with the whole retry budget spent inside those ten seconds.
    """
    ALWAYS_FLAKY.add("alpha")
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    assert _claim(db)
    assert db.run(run_id).status == "queued"

    wait_until = db.next_attempt_at(run_id)
    assert wait_until is not None, "a requeued run was left immediately re-claimable"
    assert wait_until > datetime.now(UTC), "the backoff is already in the past"

    assert _claim(db) is None, "the run was claimed before its backoff elapsed"

    # And the wait is the only thing holding it: clearing it makes the run claimable at once.
    db.clear_backoff(run_id)
    assert _claim(db) is not None


def test_the_retry_budget_still_ends_the_run_after_max_attempts_failures(db: Database) -> None:
    """The other half: separating the counters must not make the ceiling unreachable."""
    ALWAYS_FLAKY.add("alpha")
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    for expected in (1, 2, 3):
        assert _claim(db)
        assert db.run(run_id).failures == expected
        # This test is about the ceiling, not the pacing, so the backoff is stepped over
        # rather than slept through. The pacing has its own test above.
        db.clear_backoff(run_id)

    record = db.run(run_id)
    assert record.status == "failed", "max_attempts=3 should stop after the third failure"
    assert record.error_class == "transient"
    assert _claim(db) is None


# --------------------------------------------------------------------------------------
# overlapping workers
# --------------------------------------------------------------------------------------


def test_a_worker_whose_lease_was_stolen_stops_spending(db: Database) -> None:
    """The fencing check. A lost lease means somebody else owns this run now.

    ``FOR UPDATE SKIP LOCKED`` stops two workers claiming the same row in the same instant;
    it says nothing about a lease that expires while its holder is still working, which is
    the ordinary case. Without a fence the old worker keeps buying completions against a run
    the new worker may already have marked ``succeeded``.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()
    with transaction(db.conn) as cur:
        cur.execute(_STEAL_LEASE, ("some-other-worker", str(run_id)))

    provider = EchoProvider()
    ctx = RunContext(
        run_id=run_id,
        tenant_id="worker",
        input={},
        conn=db.conn,
        provider=provider,
        worker_id=WORKER_ID,
    )

    with pytest.raises(PermanentError, match="no longer leased"):
        ctx.llm("answer", "answer the payload", {"a": 1})

    assert sum(provider.call_counts.values()) == 0, "a worker spent against a run it had lost"
    assert db.count_llm_calls(run_id) == 0


def test_two_overlapping_workers_do_not_both_pay_for_the_same_call(
    db: Database,
    database_url: str,
) -> None:
    """The ledger has to be consulted *inside* the lock, not just before it.

    Two workers, two connections, one step. The first worker is parked inside
    ``provider.complete()`` — holding the run's row lock, with its ledger row written but
    not committed — while the second worker runs its *free* ledger lookup. That lookup
    therefore misses, correctly and unavoidably, and from that moment the only thing standing
    between this run and two invoices for one completion is the second lookup taken behind
    the row lock. What the second worker does next is the whole test: re-read the ledger and
    replay, or act on a decision that is now stale and buy the same completion again.

    The interleaving is forced with events rather than timed with sleeps. The version this
    replaces gave the second worker a 0.25s head start into a 1.0s provider latency and hoped
    for the best; on a loaded machine that margin evaporates, the free lookup hits, and the
    test passes without ever exercising the in-lock lookup — green for the wrong reason,
    which is the only failure mode worse than red. Here the free lookup's result is asserted
    directly, so the window cannot be missed silently.
    """
    run_id = slow_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()

    # Shared, so `call_counts` is the count of *provider* calls across both workers.
    entered = threading.Event()
    release = threading.Event()
    provider = GatedEchoProvider(entered=entered, release=release)
    failures: list[BaseException] = []

    # Both contexts are built here, on connections this thread never uses again, so the main
    # thread can observe the second worker's lookups while the worker threads own the
    # connections exclusively.
    #
    # No worker_id on either: neither context holds a lease, which is what keeps the test
    # about the ledger. A fenced context would abort on `locked_by` long before it reached
    # the lookup this test is about.
    first_conn = connect(database_url)
    second_conn = connect(database_url)
    first_ctx = RunContext(
        run_id=run_id,
        tenant_id="worker",
        input={"name": "alpha"},
        conn=first_conn,
        provider=provider,
    )
    second_ctx = ObservingContext(
        run_id=run_id,
        tenant_id="worker",
        input={"name": "alpha"},
        conn=second_conn,
        provider=provider,
    )

    def execute(ctx: RunContext) -> None:
        try:
            slow_pipeline.execute(ctx)
        except BaseException as exc:  # re-raised in the main thread, below
            failures.append(exc)

    first_worker = threading.Thread(target=execute, args=(first_ctx,), daemon=True)
    second_worker = threading.Thread(target=execute, args=(second_ctx,), daemon=True)

    try:
        first_worker.start()
        assert entered.wait(timeout=30), "the first worker never reached the provider call"

        # The first worker is now inside `complete()`, holding the run's row lock with an
        # uncommitted ledger row. Everything the second worker does from here happens inside
        # that window, by construction rather than by timing.
        second_worker.start()
        assert second_ctx.free_lookup_done.wait(timeout=30), "the free lookup never ran"
        assert second_ctx.free_lookup_result is None, (
            "the second worker's free ledger lookup hit, so this run never entered the "
            "window the in-lock lookup exists for and the test would prove nothing"
        )

        release.set()
        for thread in (first_worker, second_worker):
            thread.join(timeout=30)
    finally:
        # Released before any assertion can abandon a thread inside the gate, or the two
        # connections leak and the next test blocks behind their locks.
        release.set()
        for thread in (first_worker, second_worker):
            if thread.is_alive():
                thread.join(timeout=30)
        first_conn.close()
        second_conn.close()

    assert not failures, f"worker raised: {failures[0]!r}"
    assert all(not thread.is_alive() for thread in (first_worker, second_worker))

    billed = db.count_llm_calls(run_id, cached=False)
    assert billed == 1, (
        f"two overlapping workers billed {billed} calls for one step; the second one "
        "decided 'cache miss' before it took the lock and never revisited that decision"
    )
    assert sum(provider.call_counts.values()) == 1, "the provider was asked twice"
    assert db.count_llm_calls(run_id, cached=True) == 1, "the absorbed retry left no evidence"


def test_a_displaced_worker_cannot_overwrite_the_lease_holders_checkpoint(db: Database) -> None:
    """The fence on ``steps``, which for a long time did not exist.

    The worker's two verdict statements have always been guarded ``WHERE id=%s AND
    locked_by=%s``, so a displaced worker could not overwrite the *status* of a run somebody
    else owns. Every write to ``steps`` was unguarded and keyed on ``(run_id, name)`` alone,
    and ``worker_id`` was consulted in exactly one other place — the fencing check reached
    only from a billed call. The fence therefore stopped a displaced worker *spending money*
    and stopped nothing else: it could still walk into the current holder's committed
    checkpoint, upsert it back to ``running`` with ``output=NULL``, and then record it as
    ``failed`` on the way out. The holder, still executing, would find its own completed step
    gone and re-execute it.

    So the assertion is not only that the write raises. It is that the holder's stored output
    is still there afterwards, byte for byte, which is the thing the raise exists to protect.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()
    _take_lease(db, run_id, HOLDER_ID)

    ctx = RunContext(
        run_id=run_id,
        tenant_id="worker",
        input={},
        conn=db.conn,
        provider=EchoProvider(),
        worker_id=HOLDER_ID,
    )
    assert ctx.step("work", checkpointable, "committed-by-the-holder") == {
        "value": "committed-by-the-holder"
    }

    # The lease moves on: a sweep, a longer-than-expected step, an operator forcing a
    # re-claim. Nothing tells the holder, which is exactly the situation the fence is for.
    with transaction(db.conn) as cur:
        cur.execute(_STEAL_LEASE, (THIEF_ID, str(run_id)))

    # Different arguments, so the checkpoint does not match and the step genuinely tries to
    # re-execute and overwrite. Calling it with the same arguments would be replayed from the
    # checkpoint and would never reach a write at all — a test that proved nothing.
    with pytest.raises(PermanentError, match="no longer leased"):
        ctx.step("work", checkpointable, "written-by-the-displaced-worker")

    survivors = db.steps(run_id)
    assert len(survivors) == 1, "the displaced worker wrote a second step row"
    assert survivors[0].status == "succeeded", (
        "the displaced worker reset the current holder's committed checkpoint to 'running'; "
        "the holder will now re-execute a step it had already finished and paid for"
    )
    assert survivors[0].output == {"value": "committed-by-the-holder"}


def test_finishing_a_step_renews_the_lease_so_a_working_run_is_not_stolen(db: Database) -> None:
    """Progress, not wall time, is what holds a lease.

    The lease used to be stamped once at claim time and never refreshed, so a run whose steps
    legitimately outlive ``--lease-seconds`` was stolen, re-stolen, and burned an execution
    every time — capping a run's total runtime at ``max_attempts x lease_seconds`` no matter
    how much progress it was making, and re-running from the last checkpoint on every steal.

    The step below ages its own lease past expiry while it runs, which is what a genuinely
    slow step does to the clock. Completing it has to put the lease back in the future, and
    the proof is the claim query itself: a real worker polling immediately afterwards must
    find nothing to steal. Asserting on the timestamp alone would pass against a renewal that
    moved ``locked_at`` by a millisecond and left the run just as stealable.
    """
    run_id = flaky_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="alpha")
    db.conn.commit()
    _take_lease(db, run_id, HOLDER_ID)

    ctx = RunContext(
        run_id=run_id,
        tenant_id="worker",
        input={},
        conn=db.conn,
        provider=EchoProvider(),
        worker_id=HOLDER_ID,
    )
    aged: list[datetime] = []

    def work_for_longer_than_the_lease() -> dict[str, Any]:
        """A step body that outlives the lease, simulated by backdating it rather than slept.

        Sleeping out a real lease would make this test take a minute and would still be a
        race. Backdating ``locked_at`` produces the identical row state deterministically —
        the run has been silent for an hour as far as the claim query can tell.
        """
        _expire_lease(db, run_id)
        _, backdated = _lease(db, run_id)
        assert backdated is not None
        aged.append(backdated)
        return {"done": True}

    assert ctx.step("long_step", work_for_longer_than_the_lease) == {"done": True}

    holder, renewed = _lease(db, run_id)
    assert holder == HOLDER_ID, "renewal must not change who holds the lease"
    assert renewed is not None
    assert renewed > aged[0], (
        "completing a step left `locked_at` where the slow step had aged it to, so the lease "
        "tracks wall time rather than progress"
    )
    assert datetime.now(UTC) - renewed < timedelta(seconds=30)

    assert _claim(db) is None, (
        "a run that had just committed a step was stolen; a pipeline whose steps outlive "
        "--lease-seconds can never finish, however much progress it makes"
    )
    assert db.count_steps(run_id, status="succeeded") == 1


# --------------------------------------------------------------------------------------
# a pool of workers on one queue
# --------------------------------------------------------------------------------------

#: Threads, and claims per thread. Their product is the number of runs enqueued, so every
#: ``run_once`` call below has exactly one run available to it — which makes a ``None``
#: return unambiguous evidence that a claim was *lost*, not that the queue ran dry.
_RACERS = 4
_CLAIMS_EACH = 3

#: The row every worker's claim query will pick first, taken with a plain ``FOR UPDATE`` —
#: no ``SKIP LOCKED``, because this connection is standing in for whatever else might be
#: holding a row: another claim mid-flight, a billed call, an operator in psql.
_LOCK_HEAD_OF_QUEUE = """
    SELECT id FROM runs WHERE status = 'queued' ORDER BY created_at FOR UPDATE LIMIT 1
"""

#: Backends currently blocked on a lock. Not a condition the test waits *for* — it is the
#: diagnosis printed when the pool fails to make progress, so a red run says "three workers
#: are queued behind one row" instead of "timed out".
_COUNT_LOCK_WAITERS = """
    SELECT count(*) AS n FROM pg_stat_activity
    WHERE datname = current_database() AND wait_event_type = 'Lock'
"""


def _lock_head_of_queue(conn: psycopg.Connection[DictRow]) -> UUID:
    """Lock the run a claim query would take first, and hold it until the caller commits.

    The transaction is deliberately left open. ``connect`` gives a non-autocommit
    connection, so the row lock this takes lives until the caller rolls back — which is the
    whole obstruction.
    """
    with conn.cursor() as cur:
        cur.execute(_LOCK_HEAD_OF_QUEUE)
        row = cur.fetchone()
    assert row is not None, "precondition: the queue is not empty"
    head: UUID = row["id"]
    return head


def _release_round(start_line: threading.Barrier) -> None:
    """Let one round of claims begin, tolerating a barrier a failed worker has aborted.

    A broken barrier means a worker thread raised; the failure it recorded is a far better
    error message than ``BrokenBarrierError`` from the main thread, so this gets out of the
    way and lets the assertions report it.
    """
    with suppress(threading.BrokenBarrierError):
        start_line.wait(timeout=60)


def _await_round_one(
    db: Database,
    round_one_done: list[threading.Event],
    timeout: float = 30.0,
) -> None:
    """Require the whole pool to finish its first claim *while the head row is still locked*.

    This is the assertion, not merely a wait. ``SKIP LOCKED`` buys exactly one thing — a
    worker's claim query steps over a row somebody else is holding and takes the next one
    instead of waiting for it — and the only way to observe that is to hold a row and check
    that the pool makes progress anyway.

    Without the clause every worker queues behind the held row and nothing here completes
    until the obstruction is released, which is the pool of ten workers with an effective
    concurrency of one that the module docstring warns about. It is not a *wrong* answer,
    which is why no assertion about statuses or ledger rows can catch it; it is a queue that
    drains at one worker's pace while N-1 workers sit idle, and it fails silently forever.

    Returning early on success keeps the correct path fast; the timeout is only ever paid by
    a broken one, and it reports how many backends are stuck so the failure names its cause.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(event.is_set() for event in round_one_done):
            return
        # A poll, not a sleep-and-hope: the loop exits on the condition the instant it holds.
        time.sleep(0.01)

    with transaction(db.conn) as cur:
        cur.execute(_COUNT_LOCK_WAITERS)
        row = cur.fetchone()
    stuck = 0 if row is None else int(row["n"])
    finished = sum(event.is_set() for event in round_one_done)
    pytest.fail(
        f"only {finished} of {len(round_one_done)} workers claimed anything while one row "
        f"was held by another transaction, with {stuck} backend(s) blocked on a lock. The "
        "claim query is waiting on a row somebody else holds instead of stepping over it, "
        "so this pool drains the queue at the pace of a single worker."
    )


def test_a_pool_of_workers_hands_every_run_to_exactly_one_of_them(
    db: Database,
    database_url: str,
) -> None:
    """``FOR UPDATE SKIP LOCKED`` — the clause the module docstring calls load-bearing.

    Nothing in this suite ran two workers against the queue at once, so both halves of that
    clause could be deleted from ``_CLAIM_SQL`` and every test stayed green. Each half fails
    differently, and neither failure raises anything:

    * without ``FOR UPDATE``, nothing stops two workers selecting the same row. Both then
      run the outer ``UPDATE``, and in ``READ COMMITTED`` the second one re-checks only
      ``WHERE id = <that id>`` — which still matches — so both take the run, both execute it,
      and ``attempt`` jumps by two for work that was enqueued once;
    * without ``SKIP LOCKED``, a worker whose ``SELECT`` lands on a locked row *waits* for it
      rather than moving on. Every claim is still eventually correct, which is exactly why no
      assertion about statuses or ledger rows can see the problem: a pool of ten workers
      simply drains the queue at the pace of one, with nine of them blocked on a lock and
      nothing anywhere reporting it.

    So the test drives ``_RACERS`` real ``run_once`` calls from separate connections and
    threads, in lockstep rounds, against exactly as many runs as there are calls — and
    manufactures the contention rather than hoping for it. A barrier alone is not enough: the
    claim is one indexed statement, and four threads released together finished twelve claims
    without ever landing on each other, so the whole test passed against a build with
    ``SKIP LOCKED`` deleted. A fifth connection therefore takes a plain ``FOR UPDATE`` on the
    head-of-queue row and holds it across the first round, which is the state the clause
    exists for: a row locked by somebody, for longer than a claim should be willing to wait.

    Four assertions, covering both halves: the pool completes a full round *while that row is
    held* (it skipped rather than queued), no run id comes back twice and no run is executed
    more than once (nothing was double-claimed), every run ends terminal with none left on the
    queue, and the billed calls per run equal the single-worker baseline — concurrency is not
    allowed to cost money.
    """
    # The baseline is measured rather than assumed: what one worker spends on this pipeline
    # is the number a pool of them has to match, whatever it happens to be.
    baseline_id = slow_pipeline.enqueue(tenant_id="worker", conn=db.conn, name="baseline")
    db.conn.commit()
    assert _claim(db) == baseline_id
    assert db.run(baseline_id).status == "succeeded"
    baseline_billed = db.count_llm_calls(baseline_id, cached=False)
    assert baseline_billed == 1, "precondition: one billed call per run under one worker"

    db.truncate()

    queued = [
        slow_pipeline.enqueue(tenant_id="worker", conn=db.conn, name=f"racer-{n}")
        for n in range(_RACERS * _CLAIMS_EACH)
    ]
    db.conn.commit()

    # Shared so `call_counts` totals provider calls across the whole pool.
    provider = EchoProvider()
    # ``_RACERS + 1`` parties: the main thread is a participant, not a spectator. That is what
    # lets it hold the obstruction across exactly the first round — the workers cannot start
    # round two until it arrives at the barrier, and it does not arrive until it has released
    # the lock. Without that the pool could race ahead into rounds two and three while the
    # head row was still held, and run out of claimable runs for reasons this test is not
    # about.
    start_line = threading.Barrier(_RACERS + 1)
    claimed: list[list[UUID | None]] = [[] for _ in range(_RACERS)]
    round_one_done = [threading.Event() for _ in range(_RACERS)]
    failures: list[BaseException] = []

    def worker(index: int) -> None:
        connection = connect(database_url)
        try:
            for round_number in range(_CLAIMS_EACH):
                # Every thread issues its claim query within microseconds of every other
                # one, once per round. Distinct worker ids because `locked_by` is what the
                # lease fence and both verdict statements compare against.
                start_line.wait(timeout=60)
                claimed[index].append(
                    run_once(
                        connection,
                        worker_id=f"racer-{index}",
                        provider=provider,
                        lease_seconds=30,
                    )
                )
                if round_number == 0:
                    round_one_done[index].set()
        except BaseException as exc:  # re-raised in the main thread, below
            # Never leave the others waiting at a barrier the failed thread will not reach,
            # nor the main thread waiting for a round that will not complete.
            start_line.abort()
            for event in round_one_done:
                event.set()
            failures.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(_RACERS)]
    obstruction = connect(database_url)
    try:
        head_of_queue = _lock_head_of_queue(obstruction)
        assert head_of_queue in queued, "the obstruction locked a row that is not on the queue"

        for thread in threads:
            thread.start()
        _release_round(start_line)

        # Hold the head row until round one has resolved itself one way or the other: every
        # worker got past it (correct), or every worker is stuck behind it (broken). Both
        # conditions are observable and both terminate, so the lock is never released on a
        # timer — which is what keeps this a forced interleaving rather than a sleep race.
        _await_round_one(db, round_one_done)
    finally:
        # Rollback, not commit: this connection only ever read, and releasing the lock has to
        # happen even if the wait above failed, or the worker threads block forever.
        obstruction.rollback()
        obstruction.close()

    # The head row is claimable again; the remaining rounds run without an obstruction.
    for _ in range(_CLAIMS_EACH - 1):
        _release_round(start_line)

    for thread in threads:
        thread.join(timeout=120)

    assert not failures, f"worker raised: {failures[0]!r}"
    assert all(not thread.is_alive() for thread in threads), "a worker thread never finished"

    returned = [run_id for per_thread in claimed for run_id in per_thread]
    assert None not in returned, (
        f"{returned.count(None)} of {len(returned)} claims came back empty against a queue "
        "that had exactly one run per claim; a worker blocked behind another worker's row "
        "instead of skipping it, and the run it was meant to take was left on the queue"
    )
    assert len(set(returned)) == len(returned), (
        "a run was claimed by two workers at once — the claim query selected a row somebody "
        "else was already updating"
    )
    assert set(returned) == set(queued)

    for run_id in queued:
        record = db.run(run_id)
        assert record.status == "succeeded", f"run {run_id} ended {record.status}"
        assert record.attempt == 1, "a run was executed more than once"
        billed = db.count_llm_calls(run_id, cached=False)
        assert billed == baseline_billed, (
            f"run {run_id} was billed {billed} call(s) under a pool of workers against "
            f"{baseline_billed} under one; concurrency is not allowed to cost money"
        )
    assert sum(provider.call_counts.values()) == len(queued)


# --------------------------------------------------------------------------------------
# a worker that cannot serve the queue it is polling
# --------------------------------------------------------------------------------------


def test_an_unregistered_pipeline_is_handed_back_rather_than_failed(db: Database) -> None:
    """One worker started with the wrong ``--import`` must not empty a shared queue.

    An unknown pipeline name means the run is enqueued against code *this process* never
    imported. That is a fact about the worker's configuration, not about the run — but it
    used to be recorded as the run's permanent failure, so a single misconfigured worker
    walked the whole queue converting every claimable run to terminal ``failed``, faster than
    the correctly-configured workers could claim them, and with no way back short of
    re-enqueueing by hand.

    Three things have to be true of the hand-back, and each covers a way of getting it wrong:
    the status returns to ``queued`` (the run is available again), ``attempt`` is refunded (a
    claim that never became an execution must not spend the ceiling the claim query enforces,
    or a misconfigured worker still burns the run out — into the unclaimable state the sweep
    exists to rescue), and ``next_attempt_at`` is set (otherwise this worker re-claims the
    same run on its very next poll and spins on it instead of leaving it for someone else).
    """
    run_id = db.insert_run(pipeline=UNREGISTERED_PIPELINE, tenant_id="worker")

    assert _claim(db) == run_id, "the run was not claimed at all"

    record = db.run(run_id)
    assert record.status == "queued", (
        f"a run this worker could not execute was recorded as {record.status!r}; one worker "
        "with the wrong --import now destroys the queue it is polling"
    )
    assert record.attempt == 0, "the claim was refunded nowhere; this worker still burns the run"
    assert record.error_class is None
    assert record.error_detail is None
    assert record.failures == 0, "a hand-back is not a failure and must not spend the budget"

    holder, _ = _lease(db, run_id)
    assert holder is None, "the released run is still leased to the worker that gave it up"

    wait_until = db.next_attempt_at(run_id)
    assert wait_until is not None, "the release is unpaced; this worker will spin on the run"
    assert wait_until > datetime.now(UTC)
    assert _claim(db) is None, "the released run was re-claimed before its pacing elapsed"

    # And the run really is intact: once the pacing elapses it is claimable again, by this
    # worker or (in production) by one that can actually execute it.
    db.clear_backoff(run_id)
    assert _claim(db) == run_id
    assert db.run(run_id).status == "queued"


# --------------------------------------------------------------------------------------
# configuration that breaks the queue rather than slowing it down
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lease_seconds", "poll_interval", "expected"),
    [
        (0, 1.0, "lease_seconds"),
        (-1, 1.0, "lease_seconds"),
        (-3600, 1.0, "lease_seconds"),
        (60, 0.0, "poll_interval"),
        (60, -0.5, "poll_interval"),
    ],
)
def test_worker_timings_that_break_the_queue_are_refused(
    lease_seconds: int,
    poll_interval: float,
    expected: str,
) -> None:
    """Both flags have a value that breaks the system rather than degrading it.

    ``lease_seconds <= 0`` collapses the lease window to nothing: every ``running`` row
    satisfies ``locked_at < now() - interval '0'`` the instant it is stamped, so every poll of
    every worker steals the run another worker is currently executing, and a pool becomes a
    mutual-theft loop that burns each run's execution ceiling in seconds having completed no
    work. ``poll_interval <= 0`` turns the idle loop into a busy loop issuing claim queries as
    fast as the connection allows.

    Neither announces itself — the worker logs ``lease=0s poll=0.0s`` once at startup and then
    looks perfectly healthy — which is why this is a refusal at the door rather than a warning.
    The check lives in the library and not only in the CLI's argument parser, or
    ``run_forever`` stays callable with a configuration the CLI would reject.

    The message has to name the offending knob: "invalid worker configuration" sends an
    operator to read the source of the thing they are trying to start.
    """
    with pytest.raises(ValueError, match=expected):
        validate_worker_timings(lease_seconds=lease_seconds, poll_interval=poll_interval)


def test_the_smallest_usable_worker_timings_are_accepted() -> None:
    """The other side of the boundary, so the guard cannot be tightened into uselessness.

    A one-second lease and a sub-second poll are aggressive but coherent: both describe a
    worker that works, merely one tuned for a fast queue. A validator that rejected them
    would be enforcing taste rather than correctness, and would break every test and demo
    that wants a worker to react immediately.
    """
    validate_worker_timings(lease_seconds=1, poll_interval=0.001)
