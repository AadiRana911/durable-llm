"""The claim loop: turn queued rows into executed runs, without losing or duplicating work.

A worker is a process that repeatedly asks Postgres for one run it is allowed to execute,
executes it, and records the verdict. Everything difficult about that sentence is handled by
the single ``UPDATE ... FROM (SELECT ... FOR UPDATE SKIP LOCKED)`` statement below, so there
is no broker, no heartbeat protocol and no scheduler process to operate.

Two failure modes drive the design:

*The worker dies mid-run.* ``SIGKILL``, an OOM kill, a spot instance reclaimed — no ``finally``
block runs, no status is written, and the run sits in ``running`` forever. A **lease** fixes
this: claiming stamps ``locked_by``/``locked_at``, and the claim query is willing to steal any
``running`` row whose stamp is older than ``lease_seconds``. Recovery therefore needs nothing
but another worker starting up. Note what the lease is *not*: it is not a guarantee that the
first worker is dead, only that it has been silent for longer than the lease. That is why
resumption safety lives in ``context.py`` (checkpointed steps plus the ``llm_calls``
idempotency ledger) rather than in the assumption that only one process is ever running.

*Several workers run at once.* ``FOR UPDATE SKIP LOCKED`` makes concurrent claims safe:
each worker's ``SELECT`` locks the row it picks, and any other worker's ``SELECT`` skips
locked rows instead of blocking on them. Without ``SKIP LOCKED``, ten workers serialise
behind the same head-of-queue row and the pool has an effective concurrency of one.

The other half of this module is the verdict. When a run raises, the exception is classified
(:func:`durable_llm.errors.classify`), the class is looked up in
:data:`durable_llm.errors.DISPOSITIONS`, and the table — not an ``if`` chain here — decides
whether the run goes back to ``queued`` or lands in a terminal status with
``error_class``/``error_detail`` recorded.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from durable_llm.context import RunContext
from durable_llm.db import transaction
from durable_llm.errors import (
    DISPOSITIONS,
    IN_STEP_RETRY_CLASSES,
    MAX_BACKOFF_SECONDS,
    ErrorClass,
    PermanentError,
    backoff_delay,
    classify,
    error_detail,
)
from durable_llm.registry import get as get_pipeline
from durable_llm.types import RunRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import FrameType
    from uuid import UUID

    import psycopg
    from psycopg.rows import DictRow

    from durable_llm.pipeline import Pipeline
    from durable_llm.providers.base import Provider

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_SWEEP_INTERVAL",
    "ConnectionLost",
    "default_worker_id",
    "resolve_default_model",
    "run_forever",
    "run_once",
    "validate_worker_timings",
]

logger = logging.getLogger("durable_llm.worker")

DEFAULT_LEASE_SECONDS = 60
"""How long a claim is trusted before another worker may steal the run.

Checkpoint transactions in ``context.py`` renew the lease when a step starts and when it
successfully finishes, tying ownership to forward progress without a timer heartbeat. There
is no renewal while a step body itself runs, so raise this if one step is legitimately slower
than a minute. The cost is a longer wait before a genuinely crashed run is picked back up.
"""

DEFAULT_POLL_INTERVAL = 1.0
"""Seconds an idle worker waits before asking for work again. A poll, not a queue: at this
scale one indexed query per second per worker is free, and the operational simplicity of
having no message broker is worth far more than the latency."""

DEFAULT_SWEEP_INTERVAL = float(DEFAULT_LEASE_SECONDS)
"""How often ``run_forever`` sweeps for stranded runs.

The sweep used to run before every single claim, which at the default one-second poll meant
one extra write transaction per worker per second forever, to find nothing almost every
time. Stranding is a rare, slow event, so a fixed one-minute interval is sufficient and, at
any pool size, an order of magnitude less load. ``run_once`` still sweeps on every call by
default, because a caller invoking it directly has no loop to pace it from.
"""


# ======================================================================================
#  SQL
# ======================================================================================

# The claim. One statement does all four jobs: pick a run, take it, prove nobody else took
# it, and hand back the whole row.
#
#   * `status='queued'` is the normal path.
#   * `status='running' AND locked_at < now() - lease` is the recovery path: this is the
#     clause that reclaims a run whose worker was SIGKILLed. Nothing else in the system
#     notices that a process died, and nothing needs to.
#   * FOR UPDATE SKIP LOCKED is what makes concurrent workers safe. The inner SELECT locks
#     the row it returns, and a second worker running the identical query skips that locked
#     row and takes the next one instead of blocking behind it. Drop SKIP LOCKED and N
#     workers queue up behind one row; drop FOR UPDATE and two workers can claim the same
#     run in the window between SELECT and UPDATE.
#   * attempt=attempt+1 happens at claim time, not at failure time, so a worker that dies
#     without writing anything still burns an attempt. Otherwise a run that reliably kills
#     its worker is retried forever.
#   * `attempt < max_attempts` is what makes the previous line mean something. Incrementing
#     a counter nobody reads bounds nothing: the ceiling has to be enforced *here*, because
#     a SIGKILLed worker never reaches the code that writes a verdict. `max_attempts` is a
#     column on the run (copied from the pipeline at enqueue time) rather than a parameter,
#     because different pipelines in the same queue deserve different ceilings.
#   * This bounds *executions*, not failures. The retry budget for a run that actually
#     failed is `runs.failures`, consulted in _record_failure — see migrations/0002.
#   * `next_attempt_at` is the run-level backoff, added in migrations/0003. A requeued run
#     used to be claimable on the very next poll, which meant a provider having a bad ten
#     seconds got answered with the entire retry budget spent inside those ten seconds. NULL
#     means "now" and is what every row not currently backing off carries.
_CLAIM_SQL = """
UPDATE runs SET status='running', locked_by=%s, locked_at=now(), attempt=attempt+1,
    next_attempt_at=NULL, updated_at=now()
WHERE id = (
    SELECT id FROM runs
    WHERE attempt < max_attempts
      AND (next_attempt_at IS NULL OR next_attempt_at <= now())
      AND (status='queued'
           OR (status='running' AND locked_at < now() - make_interval(secs => %s)))
    ORDER BY created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING *;
"""

# The companion to the ceiling above. Without it, a run that reached the ceiling without ever
# recording a verdict would sit there forever: invisible to the claim query, and
# indistinguishable in `durable-llm runs list` from one that is still being worked on. This turns
# it into a terminal row that says what happened, which is the whole difference between a
# stranded run and a diagnosed one.
#
# Two shapes of stranding, not one, and the second is the one that bit hardest:
#
#   * status='running' — the worker died. Nothing classified the failure because nothing
#     survived to classify it. The lease check is what keeps this from firing on a run that
#     is genuinely still executing its final allowed attempt.
#   * status='queued' — the run was *requeued* at or past its ceiling. `_record_failure`
#     used to decide that from `failures` alone, so lease churn (which burns `attempt`
#     without burning `failures`) could requeue a run the claim query would never take back.
#     No lease check applies here: a queued row is by definition held by nobody, so there is
#     nothing to wait out. `_record_failure` no longer creates this state; this clause is
#     what rescues the rows created before it stopped.
#
# `FOR UPDATE SKIP LOCKED` in the subquery is not an optimisation, it is the difference
# between a poll loop and a stall. As a plain UPDATE this statement blocked on any matching
# row whose lock was held — and `_billed_call` holds the run's row lock across the provider
# network call, so one worker inside a slow completion stalled every other worker's poll for
# the length of that call. A locked row is visibly being worked on; skipping it is both
# faster and more correct, and it is what context.py's own commentary already claimed
# happened here.
#
# LIMIT bounds the write: a database that has accumulated thousands of stranded rows should
# drain them over several polls rather than take one enormous lock-heavy transaction.
_SWEEP_EXHAUSTED_SQL = """
UPDATE runs
SET status='failed',
    error_class='permanent',
    error_detail=CASE WHEN status='queued' THEN %s ELSE %s END,
    locked_by=NULL,
    locked_at=NULL,
    next_attempt_at=NULL,
    updated_at=now()
WHERE id IN (
    SELECT id FROM runs
    WHERE attempt >= max_attempts
      AND (status='queued'
           OR (status='running' AND locked_at < now() - make_interval(secs => %s)))
    FOR UPDATE SKIP LOCKED
    LIMIT %s
)
RETURNING id, error_detail
"""

_SWEEP_LIMIT = 100
"""Rows one sweep may terminate. See _SWEEP_EXHAUSTED_SQL."""

_SWEEP_DETAIL_DIED = (
    "PermanentError: the worker died without recording a verdict on every allowed attempt "
    "(runs.attempt reached runs.max_attempts). Nothing classified this failure because "
    "nothing survived to classify it — check the worker logs for the run id, fix the cause, "
    "then re-enqueue."
)

_SWEEP_DETAIL_STRANDED = (
    "PermanentError: this run was requeued after it had already used every allowed "
    "execution (runs.attempt reached runs.max_attempts), which left it queued but "
    "unclaimable. It is being failed so it stops reading as work in progress. Its recorded "
    "failures are below its retry budget, so the cause is lease churn rather than the "
    "pipeline: raise --lease-seconds or max_attempts and re-enqueue."
)

# Hands a run back when *this* worker cannot execute it but another one might — the
# unimportable-pipeline case. Deliberately not a failure: the run is fine, this process is
# the thing that is misconfigured, and writing a terminal status would destroy a queue over a
# deployment mistake.
#
# `attempt` is decremented because the claim never became an execution. Leaving it
# incremented would let a worker with the wrong --import burn a run's entire execution
# ceiling without ever calling it, arriving at exactly the stranded state above.
#
# `next_attempt_at` paces the release: without it this worker re-claims the same run on its
# very next poll and spins on it. With it, the run is parked long enough for a correctly
# configured worker to take it.
_RELEASE_SQL = """
UPDATE runs
SET status='queued',
    locked_by=NULL,
    locked_at=NULL,
    attempt=GREATEST(attempt - 1, 0),
    next_attempt_at=now() + make_interval(secs => %s),
    updated_at=now()
WHERE id=%s AND locked_by=%s
"""

# Both verdict statements are guarded by `locked_by = %s`: if our lease expired and another
# worker has already claimed this run, we must not overwrite its status with ours. rowcount
# tells us that happened.
_SUCCEED_SQL = """
UPDATE runs
SET status='succeeded',
    output=%s,
    error_class=NULL,
    error_detail=NULL,
    locked_by=NULL,
    locked_at=NULL,
    updated_at=now()
WHERE id=%s AND locked_by=%s
"""

# `failures = failures + 1` lives here and nowhere else. It is the counter the retry budget
# is measured against, and incrementing it only when a verdict is actually recorded is what
# keeps a lease expiry from spending an attempt the run never used.
#
# `next_attempt_at` carries the run-level backoff on a requeue and NULL on a terminal
# verdict. It is computed in Python from the same `errors.backoff_delay` the in-step loop
# uses, so the two levels of retry pace themselves by one function rather than two.
_VERDICT_SQL = """
UPDATE runs
SET status=%s,
    error_class=%s,
    error_detail=%s,
    failures=failures + 1,
    locked_by=NULL,
    locked_at=NULL,
    next_attempt_at=%s,
    updated_at=now()
WHERE id=%s AND locked_by=%s
"""


# ======================================================================================
#  identity and configuration
# ======================================================================================


def default_worker_id() -> str:
    """Stable, human-readable identity for this process: ``hostname/pid``.

    It lands in ``runs.locked_by``, so when a run is stuck the first diagnostic question —
    *which box, which process?* — is answered by the row itself. Stable across the life of
    the process (unlike a random UUID per claim) so that a worker can recognise its own
    lease, and unique across hosts and restarts because a PID is only unique per host and
    only until it is recycled.
    """
    return f"{socket.gethostname()}/{os.getpid()}"


def resolve_default_model(provider: Provider, override: str | None = None) -> str:
    """Decide which model ``ctx.llm()`` calls use when they do not name one.

    Order: the caller's ``--model``, then whatever the provider instance was configured
    with. There is no third step, and that is the point — this module used to carry a
    ``DEFAULT_MODELS`` table mapping ``"anthropic"`` to a model id, which duplicated
    :data:`durable_llm.providers.anthropic.DEFAULT_MODEL` across the provider seam and put the
    core in the business of knowing one adapter's model names. A provider's default model is
    the adapter's business; both bundled adapters expose it as ``.model``.

    Raises:
        PermanentError: The provider names no model. Failing here — before a run is claimed
            — is deliberate: the alternative is claiming a run and failing it for a
            configuration mistake that has nothing to do with it.
    """
    if override:
        return override

    for attribute in ("default_model", "model"):
        configured = getattr(provider, attribute, None)
        if isinstance(configured, str) and configured:
            return configured

    name = getattr(provider, "name", "")
    raise PermanentError(
        f"no default model is known for provider {name or provider!r}; pass "
        "default_model=... (or --model on the CLI), or give the provider a .model attribute"
    )


# ======================================================================================
#  one run
# ======================================================================================


def run_once(
    conn: psycopg.Connection[DictRow],
    *,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    provider: Provider,
    default_model: str | None = None,
    sweep: bool = True,
) -> UUID | None:
    """Claim at most one run, execute it, and record the verdict.

    Returns the id of the run that was processed, or ``None`` when nothing was claimable.
    Both outcomes are normal; ``None`` simply means the queue is empty and every running
    run is still inside its lease.

    This function does not raise for pipeline failures — a failed run is a recorded
    outcome, not an exception for the caller to handle. It *does* propagate infrastructure
    failures (a dead connection, for instance), because those are not something a status
    column can express.

    Args:
        sweep: Whether to look for stranded runs first. ``run_forever`` pages this down to
            once per fixed sweep interval; a direct caller has no loop to pace it from, so
            the default is every call.
    """
    # Resolved before claiming: a misconfigured provider should not cost a run an attempt.
    model = resolve_default_model(provider, default_model)

    # Before claiming, not after: a run past its execution ceiling is not claimable, so if
    # it is the only work in the queue then claiming first would report "nothing to do" and
    # leave it stranded forever.
    if sweep:
        _sweep_exhausted(conn, lease_seconds=lease_seconds)

    run = _claim(conn, worker_id=worker_id, lease_seconds=lease_seconds)
    if run is None:
        return None

    logger.info(
        "claimed run %s (pipeline=%s tenant=%s worker=%s attempt=%d/%d failures=%d)",
        run.id,
        run.pipeline,
        run.tenant_id,
        worker_id,
        run.attempt,
        run.max_attempts,
        run.failures,
    )

    # Resolved before the try below, because "this process cannot run this pipeline" is a
    # fact about the *worker*, not about the run, and must not be recorded as the run's
    # failure. A worker started with the wrong --import claims every run in the queue in
    # turn; treating each one as a permanent failure converted an entire shared queue to
    # terminal `failed` over one deployment mistake.
    try:
        pipeline = get_pipeline(run.pipeline)
    except KeyError as exc:
        _release_claim(conn, run=run, worker_id=worker_id, lease_seconds=lease_seconds, exc=exc)
        return run.id

    try:
        ctx = RunContext(
            run_id=run.id,
            tenant_id=run.tenant_id,
            input=run.input,
            conn=conn,
            provider=provider,
            budget_usd=run.budget_usd,
            default_model=model,
            # The fencing token. Every billed call re-reads `runs.locked_by` under the row
            # lock it is about to spend behind, so a worker whose lease expired mid-run
            # stops rather than buying completions against a run someone else now owns.
            worker_id=worker_id,
        )
        output = pipeline.execute(ctx)
        _reject_unserialisable(run.pipeline, output)
    # Broad on purpose: classifying an arbitrary exception is this function's entire job,
    # and anything unrecognised is treated as a permanent bug rather than retried.
    except Exception as exc:
        _record_failure(conn, run=run, exc=exc, pipeline=pipeline, worker_id=worker_id)
    else:
        _record_success(conn, run=run, output=output, worker_id=worker_id)

    return run.id


def _claim(
    conn: psycopg.Connection[DictRow],
    *,
    worker_id: str,
    lease_seconds: int,
) -> RunRecord | None:
    """Run the claim query and commit, so the ``running`` stamp is immediately durable.

    Committing here rather than holding the transaction open for the whole run is the
    point: the row lock is released at once (a run that takes four minutes must not block
    the claim query for four minutes), and the lease — a timestamp any worker can read —
    becomes the only ownership signal.
    """
    with transaction(conn) as cur:
        # float() so the parameter matches make_interval's double precision `secs`
        # argument exactly, instead of relying on an implicit integer cast.
        cur.execute(_CLAIM_SQL, (worker_id, float(lease_seconds)))
        row = cur.fetchone()
    return None if row is None else RunRecord.from_row(row)


def _sweep_exhausted(conn: psycopg.Connection[DictRow], *, lease_seconds: int) -> None:
    """Terminate runs that used every allowed execution but never reached a terminal status.

    Two populations, both of which read as work in progress and are neither:

    * ``running`` — the worker died (OOM, SIGKILL, a reclaimed spot instance) often enough to
      exhaust ``max_attempts``. Nothing classified them, because nothing survived to classify
      them, so somebody has to say so out loud.
    * ``queued`` — the run was requeued at or past its ceiling and the claim query will never
      take it back. ``_record_failure`` no longer produces this, but rows already in that
      state have to be rescued by something.

    The lease check applies only to the first: a run on its final allowed attempt is still
    *inside* its lease while that attempt runs, and failing it out from under a live worker
    would be the same double-execution bug from the other direction. A queued row is held by
    nobody, so there is nothing to wait out.
    """
    with transaction(conn) as cur:
        cur.execute(
            _SWEEP_EXHAUSTED_SQL,
            (_SWEEP_DETAIL_STRANDED, _SWEEP_DETAIL_DIED, float(lease_seconds), _SWEEP_LIMIT),
        )
        swept = [(row["id"], row["error_detail"]) for row in cur.fetchall()]

    for run_id, detail in swept:
        cause = "requeued past its ceiling" if detail == _SWEEP_DETAIL_STRANDED else "no verdict"
        logger.error("run %s -> failed: exhausted its attempt ceiling (%s)", run_id, cause)

    if len(swept) == _SWEEP_LIMIT:
        # The sweep is capped, so a full batch means there is very likely more waiting. Say
        # so rather than letting a large backlog drain silently over many polls and look
        # like nothing is happening.
        logger.warning(
            "swept the maximum of %d stranded run(s) in one pass; more may remain",
            _SWEEP_LIMIT,
        )


def _release_claim(
    conn: psycopg.Connection[DictRow],
    *,
    run: RunRecord,
    worker_id: str,
    lease_seconds: int,
    exc: BaseException,
) -> None:
    """Hand a claimed run back because *this* worker cannot execute it.

    The only cause today is an unregistered pipeline name: the run is enqueued against code
    this process never imported. That is a property of the worker's ``--import`` set, not of
    the run, and recording it as the run's permanent failure meant one misconfigured worker
    walked the whole queue converting every claimable run to terminal ``failed``.

    Logged at ERROR because it is always an operator mistake and always needs fixing — a
    worker that cannot serve the queue it is polling will otherwise release runs quietly
    forever while the backlog grows.
    """
    with transaction(conn) as cur:
        cur.execute(_RELEASE_SQL, (float(lease_seconds), run.id, worker_id))
        released = cur.rowcount

    if not released:
        _warn_lease_lost(run, "released")
        return

    logger.error(
        "run %s requeued: this worker cannot execute pipeline %r (%s). The run is unharmed "
        "and its attempt was refunded; start a worker with the right --import, or this one "
        "will keep handing it back every %.0fs.",
        run.id,
        run.pipeline,
        exc,
        float(lease_seconds),
    )


def _record_success(
    conn: psycopg.Connection[DictRow],
    *,
    run: RunRecord,
    output: Any,
    worker_id: str,
) -> None:
    with transaction(conn) as cur:
        cur.execute(_SUCCEED_SQL, (Jsonb(output), run.id, worker_id))
        updated = cur.rowcount
    if updated:
        logger.info("run %s succeeded", run.id)
    else:
        _warn_lease_lost(run, "succeeded")


def _record_failure(
    conn: psycopg.Connection[DictRow],
    *,
    run: RunRecord,
    exc: BaseException,
    pipeline: Pipeline | None,
    worker_id: str,
) -> None:
    """Classify the failure, consult the disposition table, and write the verdict."""
    # The pipeline may have died with an open transaction (a failed statement leaves the
    # connection in an aborted state, where every subsequent statement errors). Clear it
    # before trying to write the verdict, or the verdict is lost too and the run has to
    # wait out its lease for no reason.
    with suppress(Exception):
        conn.rollback()

    error_class = classify(exc)
    disposition = DISPOSITIONS[error_class]
    detail = error_detail(exc)

    # A bug is the one failure class where the stack is the diagnosis, and it used to be the
    # one class that threw the stack away: everything went to DEBUG, which nothing ships
    # with. A retryable failure is ordinary and stays quiet; an unclassified exception —
    # PERMANENT, i.e. "we do not know what this is" — gets its traceback at ERROR, where the
    # person who has to fix it will actually see it.
    if error_class is ErrorClass.PERMANENT:
        logger.error("run %s raised an unclassified exception", run.id, exc_info=exc)
    else:
        logger.debug("run %s raised", run.id, exc_info=exc)

    # Two ceilings apply. The disposition says how many attempts this *kind* of failure
    # deserves; the run says how many attempts this *workload* deserves. The lower one
    # wins, so `@pipeline(max_attempts=1)` really means "never retry this", whatever the
    # failure was.
    #
    # The run's ceiling is read from `run.max_attempts` — the column, copied onto the row at
    # enqueue time — and not from the live `pipeline.max_attempts` this process happens to
    # have imported. Those are two different numbers the moment a deploy changes the
    # decorator, and the claim query enforces the *column*: a worker that requeued against a
    # raised in-process limit produced a row the claim query would not take back. The
    # `pipeline` argument is still accepted so a caller can pass it, but the row wins.
    limit = min(disposition.max_attempts, run.max_attempts)
    if pipeline is not None and pipeline.max_attempts < limit:
        limit = pipeline.max_attempts

    # A third rule, and the one that keeps `max_attempts` meaning one thing. Some classes
    # spend their attempt budget inside the step (malformed output gets a repair prompt, per
    # the disposition table's strategy column). By the time such a failure reaches here its
    # attempts are already gone, so re-claiming the run would silently double the number the
    # table states. See errors.IN_STEP_RETRY_CLASSES.
    exhausted_in_step = error_class in IN_STEP_RETRY_CLASSES

    # `failures`, not `attempt`. `attempt` counts claims, which includes lease-expiry
    # steals: a pipeline whose steps legitimately outlive --lease-seconds would otherwise
    # arrive at its first genuine transient error with the whole retry budget already spent,
    # and be failed terminally having never been retried once. `failures` counts only
    # recorded verdicts, and _VERDICT_SQL below is about to add this one — so the count
    # after this failure is what the ceiling has to be compared against.
    failures_after_this = run.failures + 1
    has_failure_budget = disposition.retry and not exhausted_in_step and failures_after_this < limit

    # ...and `attempt` as well, because the two counters bound different things and a requeue
    # needs *both*. `failures` says the run deserves another try; `attempt` says whether the
    # claim query will actually give it one. Requeueing on the first alone wrote rows with
    # `status='queued' AND attempt >= max_attempts`, which `_CLAIM_SQL` rejects (it gates on
    # `attempt < max_attempts`) and the sweep did not see (it only looked at `running`). The
    # run then sat in `queued` forever: no error, no terminal status, no work being done, and
    # nothing in the system able to notice. Lease churn reaches this state without a single
    # genuine failure, because every steal burns an `attempt` and no `failures`.
    can_be_claimed_again = run.attempt < run.max_attempts
    requeue = has_failure_budget and can_be_claimed_again
    status = "queued" if requeue else disposition.terminal_status

    if has_failure_budget and not can_be_claimed_again:
        # Worth its own line: the run still had retries left on paper and is being failed
        # anyway, which without an explanation looks like the retry budget being ignored.
        detail = (
            f"{detail} [durable_llm: failed with {limit - failures_after_this} retry(s) still "
            f"unspent because the run has used all {run.max_attempts} of its allowed "
            f"executions (runs.attempt). Claims include lease-expiry steals, so raise "
            f"--lease-seconds if the steps are slow, or max_attempts if they are not.]"
        )

    # The wake-up time, computed from the same jittered backoff the in-step retry loop uses
    # so the two levels cannot drift into different pacing policies. A terminal verdict
    # clears it: nothing is waiting for a row nobody will claim.
    next_attempt_at: datetime | None = None
    if requeue:
        next_attempt_at = datetime.now(UTC) + timedelta(
            seconds=backoff_delay(disposition, failures_after_this)
        )

    with transaction(conn) as cur:
        cur.execute(
            _VERDICT_SQL,
            (status, error_class.value, detail, next_attempt_at, run.id, worker_id),
        )
        updated = cur.rowcount

    if not updated:
        _warn_lease_lost(run, status)
        return

    if requeue:
        # Scheduled rather than immediately re-claimable. The wake-up time lives in
        # `runs.next_attempt_at` (migrations/0003) rather than in a sleep here, because
        # sleeping would idle the whole worker on behalf of one run while the rest of the
        # queue waits. The in-step backoff — pausing between provider calls so a 429 is not
        # answered with another 429 — still happens in context.py, with the same function.
        logger.warning(
            "run %s failed %d/%d time(s) (%s) on claim %d/%d, requeued for %s: %s",
            run.id,
            failures_after_this,
            limit,
            error_class.value,
            run.attempt,
            run.max_attempts,
            next_attempt_at.isoformat(timespec="seconds") if next_attempt_at else "now",
            detail,
        )
    else:
        logger.error(
            "run %s -> %s after %d failure(s) across %d claim(s) (%s): %s",
            run.id,
            status,
            failures_after_this,
            run.attempt,
            error_class.value,
            detail,
        )


def validate_worker_timings(*, lease_seconds: int, poll_interval: float) -> None:
    """Reject lease and poll values that make the worker misbehave rather than run slowly.

    Both flags were previously unvalidated, and both have a value that quietly breaks the
    system rather than degrading it:

    * ``lease_seconds <= 0`` collapses the lease window to nothing. Every ``running`` row
      satisfies ``locked_at < now() - interval '0'`` the instant it is stamped, so every
      poll of every worker steals the run another worker is currently executing. A pool
      turns into a continuous mutual-theft loop that burns each run's execution ceiling in
      seconds having completed no work.
    * ``poll_interval <= 0`` turns the idle loop into a busy loop issuing claim queries as
      fast as the connection allows.

    Neither announces itself: the worker logs ``lease=0s poll=0.0s`` once at startup and
    then looks perfectly healthy. Raised as ``ValueError`` so the library API refuses the
    same values the CLI refuses — the CLI's check alone would leave ``run_forever`` callable
    with a configuration the CLI would not accept.
    """
    if lease_seconds < 1:
        raise ValueError(
            f"lease_seconds must be at least 1, got {lease_seconds}. A non-positive lease "
            "makes every running run instantly stealable by every other worker."
        )
    if poll_interval <= 0:
        raise ValueError(
            f"poll_interval must be greater than 0, got {poll_interval}. A zero poll turns "
            "the idle loop into a busy loop against the database."
        )


class ConnectionLost(RuntimeError):
    """The worker's database connection died and no ``reconnect`` factory was supplied.

    Raised out of :func:`run_forever` so the process exits non-zero and a supervisor can
    restart it. The alternative — the old behaviour — was to catch the failure, sleep, and
    retry the same dead handle forever: a live process, logging an exception every poll,
    processing nothing, and reporting healthy to anything that only checks whether it is
    running.
    """


def _error_backoff(consecutive_errors: int, poll_interval: float) -> float:
    """Spacing between retries of a failing loop iteration.

    Backs off rather than retrying at the poll interval forever, because the usual cause is
    a database that is down or failing over, and a pool of workers hammering it once a
    second each is not helping it come back. Capped so recovery is still prompt.
    """
    return min(poll_interval * (2 ** min(consecutive_errors - 1, 6)), MAX_BACKOFF_SECONDS)


def _recover_connection(
    conn: psycopg.Connection[DictRow],
    consecutive_errors: int,
    reconnect: Callable[[], psycopg.Connection[DictRow]] | None,
) -> psycopg.Connection[DictRow]:
    """Deal with whatever just broke the loop, and return the connection to keep using.

    A live connection that merely had a statement fail is rolled back and reused. A
    connection that is *dead* — closed or broken, which is what a failover, an admin
    shutdown or an idle-in-transaction timeout leaves behind — cannot be rolled back at all
    (``rollback()`` itself raises), so it is replaced.

    Raises:
        ConnectionLost: The connection is dead and the caller supplied no way to build a
            new one. Exiting is the honest outcome; spinning on a dead handle is not.
    """
    logger.exception("worker loop error (%d in a row)", consecutive_errors)

    if not (conn.closed or conn.broken):
        with suppress(Exception):
            conn.rollback()
        return conn

    if reconnect is None:
        raise ConnectionLost(
            "the worker's database connection is closed and no reconnect factory was "
            "supplied; exiting so a supervisor can restart this process"
        )

    with suppress(Exception):
        conn.close()
    try:
        replacement = reconnect()
    except Exception:
        # The database is still unreachable. Hand back the dead handle: the caller backs off
        # and calls here again, so this becomes a paced, loudly-logged reconnection loop
        # rather than the silent spin on an unusable connection this replaced.
        logger.exception("could not re-establish the database connection; will retry")
        return conn
    logger.warning("database connection was lost and has been re-established")
    return replacement


def _warn_lease_lost(run: RunRecord, intended_status: str) -> None:
    """Another worker owns this run now; our verdict is stale and was not written."""
    logger.warning(
        "run %s: lease lost while executing, %r verdict discarded (another worker has "
        "claimed it); increase --lease-seconds if this recurs",
        run.id,
        intended_status,
    )


def _reject_unserialisable(pipeline_name: str, output: Any) -> None:
    """Fail the run if its output cannot be stored, instead of failing the UPDATE.

    ``runs.output`` is ``jsonb``. A pipeline returning a ``datetime``, a ``Decimal`` or a
    ``NaN`` would otherwise blow up inside the success path, leave the run in ``running``,
    and have it re-claimed on lease expiry — an infinite loop over a bug that a
    ``permanent`` failure states plainly in one row.

    ``allow_nan=False`` is the strict setting: Python's default emits the literal ``NaN``,
    which is not JSON and which ``jsonb`` rejects.
    """
    try:
        json.dumps(output, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PermanentError(
            f"pipeline {pipeline_name!r} returned output that cannot be stored as jsonb: "
            f"{exc}. Return plain JSON types (str/int/float/bool/None/list/dict)."
        ) from exc


# ======================================================================================
#  the loop
# ======================================================================================


class _ShutdownFlag:
    """Cooperative stop signal, set from a signal handler.

    A signal handler must do almost nothing — it can interrupt the main thread between any
    two bytecodes, including in the middle of a transaction. So it sets a flag, and the
    loop decides when it is safe to act on it: after the current run has been recorded,
    never in the middle of one.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._signal: int | None = None

    def __bool__(self) -> bool:
        return self._event.is_set()

    @property
    def signal_number(self) -> int | None:
        return self._signal

    def handle(self, signum: int, frame: FrameType | None) -> None:
        """Signal-handler entry point. ``frame`` is required by the C API, unused here."""
        del frame
        if self._event.is_set():
            # A second signal means the operator is not prepared to wait for the current
            # run. Restore default handling so the next one terminates the process; the
            # run stays 'running' and another worker reclaims it when the lease expires.
            logger.warning("second signal %d received, restoring default handler", signum)
            with suppress(ValueError, OSError):
                signal.signal(signum, signal.SIG_DFL)
            return
        self._signal = signum
        self._event.set()

    def wait(self, seconds: float) -> None:
        """Sleep, waking early if a stop was requested.

        Sliced rather than one long wait: a signal handler runs during the sleep but does
        not interrupt it (PEP 475 resumes the call with the remaining timeout), so
        responsiveness comes from re-checking the flag often.
        """
        deadline = time.monotonic() + seconds
        while not self._event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._event.wait(min(remaining, 0.2))


@contextmanager
def _graceful_shutdown(
    signums: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
) -> Iterator[_ShutdownFlag]:
    """Install cooperative handlers for SIGTERM/SIGINT and restore them afterwards.

    SIGTERM is what a container runtime, systemd or ``kill`` sends first, and handling it
    is the difference between a rolling deploy that finishes its work and one that abandons
    a run for the length of a lease. SIGKILL cannot be handled at all — that is precisely
    what the chaos test uses, and the lease is the answer to it.

    Restoring the previous handlers matters because tests call the worker in-process.
    """
    flag = _ShutdownFlag()
    installed: list[tuple[int, Any]] = []
    for signum in signums:
        try:
            installed.append((signum, signal.signal(signum, flag.handle)))
        except ValueError:
            # Only the main thread may install handlers; a worker running inside a test
            # thread simply does not get graceful shutdown.
            logger.debug("cannot install handler for signal %d off the main thread", signum)
    try:
        yield flag
    finally:
        for signum, previous in installed:
            with suppress(ValueError, OSError, TypeError):
                signal.signal(signum, previous)


def run_forever(
    conn: psycopg.Connection[DictRow],
    *,
    provider: Provider,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    default_model: str | None = None,
    max_runs: int | None = None,
    reconnect: Callable[[], psycopg.Connection[DictRow]] | None = None,
) -> None:
    """Claim and execute runs until SIGTERM/SIGINT, or until ``max_runs`` are processed.

    On SIGTERM the loop finishes the run it holds, writes the verdict, and returns so the
    caller can close the connection.

    Args:
        reconnect: How to build a replacement connection when this one dies. The loop cannot
            derive it — ``conn.info.dsn`` withholds the password, by design — so the caller
            that knew the DSN has to supply it. When omitted, a dead connection raises
            :class:`ConnectionLost` instead, so the process exits and a supervisor restarts
            it. What the loop will *not* do is what it used to: catch the error, sleep, and
            retry the same dead handle forever while reporting healthy.
    """
    validate_worker_timings(lease_seconds=lease_seconds, poll_interval=poll_interval)

    identity = worker_id or default_worker_id()
    processed = 0
    # Resolved once, at startup, so a provider that names no model is a crash on the first
    # line rather than an exception on every iteration of the loop below forever.
    model = resolve_default_model(provider, default_model)

    # `run_once` sweeps for stranded runs on every call, which is right for a direct caller
    # and wasteful once per second forever. Stranded rows are slow-path maintenance, so a
    # fixed minute is frequent enough. Monotonic, because this is a duration and a wall clock
    # that steps backwards would park the sweep until it caught up.
    next_sweep = 0.0
    consecutive_errors = 0

    with _graceful_shutdown() as shutdown:
        logger.info(
            "worker %s started (provider=%s model=%s lease=%ds poll=%.1fs)",
            identity,
            getattr(provider, "name", provider),
            model,
            lease_seconds,
            poll_interval,
        )
        while not shutdown:
            now = time.monotonic()
            due_for_sweep = now >= next_sweep
            try:
                run_id = run_once(
                    conn,
                    worker_id=identity,
                    lease_seconds=lease_seconds,
                    provider=provider,
                    default_model=model,
                    sweep=due_for_sweep,
                )
            except Exception:
                # Pipeline failures never reach here; this is the database going away, and
                # a worker that exits on a failover is a worker that needs a babysitter.
                consecutive_errors += 1
                conn = _recover_connection(conn, consecutive_errors, reconnect)
                shutdown.wait(_error_backoff(consecutive_errors, poll_interval))
                continue

            consecutive_errors = 0
            if due_for_sweep:
                next_sweep = now + DEFAULT_SWEEP_INTERVAL

            if run_id is None:
                shutdown.wait(poll_interval)
                continue

            processed += 1
            if max_runs is not None and processed >= max_runs:
                logger.info("worker %s reached max_runs=%d", identity, max_runs)
                break

        if shutdown:
            logger.info(
                "worker %s: signal %s received, shutting down cleanly",
                identity,
                shutdown.signal_number,
            )

    logger.info("worker %s stopped after %d run(s)", identity, processed)
