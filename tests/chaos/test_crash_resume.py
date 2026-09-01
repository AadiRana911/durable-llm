"""Kill a worker mid-run with SIGKILL and prove the bill does not double.

This is the test the whole project exists to make possible, so it is worth being precise
about what it does and does not demonstrate.

**The experiment.** Run the twelve-step example cleanly once and record two things: the
output, and how many *billed* provider calls it took (``llm_calls`` rows with
``cached = false``). Then wipe the tables, enqueue the same work again, start a real
``llmrun worker`` subprocess, wait until roughly half the steps have committed, and
``SIGKILL`` its process group. Not ``terminate()``, not a graceful shutdown, not a
context manager that gets a chance to clean up — signal 9 to the group, which is what an
OOM killer, a spot-instance reclaim and ``kubectl delete pod --force`` all look like from
inside the process. Then start a second worker and let it finish.

**The claim.** The second worker must produce byte-identical output *and* the run must
have cost exactly what the uninterrupted run cost. Not "about the same". Equal. That is
the assertion a naive implementation fails: without a checkpoint table the resumed run
redoes every step, and without an idempotency ledger keyed by a *stable* hash it redoes
every model call, and neither failure is visible anywhere except the invoice.

**What it does not prove.** There is a window of a few microseconds between a provider
returning a response and the transaction recording it committing. A kill landing exactly
there loses the record of a call the vendor has already billed, and the resumed run makes
it again. No amount of application code closes that window — it would need a distributed
transaction with the vendor's billing system. llmrun shrinks it to a single round trip and
is honest about the rest.

The suite is POSIX-only (process groups, ``SIGKILL``) and marked ``chaos`` so ordinary
``make test`` runs skip it; ``make chaos`` and CI run it.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from llmrun.types import RunRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from conftest import Database

logger = logging.getLogger("llmrun.tests.chaos")

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        os.name != "posix",
        reason="the crash suite needs POSIX process groups and SIGKILL",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "examples" / "summarise_thread.py"

#: Passed to ``--import`` exactly as the Makefile and the README pass it, so this test
#: exercises the documented invocation rather than a private one.
EXAMPLE_IMPORT_ARG = "examples/summarise_thread.py"
PIPELINE_NAME = "summarise_thread"
THREAD_ID = "thread-chaos-42"

BASELINE_TENANT = "chaos-baseline"
CRASH_TENANT = "chaos-crash"

#: Short lease so the crashed run is reclaimable in seconds rather than the default minute.
#: A production deployment wants this an order of magnitude longer than its slowest step.
LEASE_SECONDS = 2

#: Artificial per-step delay for the run that gets killed. Twelve echo steps otherwise
#: complete faster than another process can watch them and pull the trigger.
#:
#: It also sets how wide the window in :func:`_wait_for_crash_window` is: the example's
#: schema validators pace too, and llmrun calls the validator twice between committing a
#: provider call and checkpointing its step, so the window is about twice this value. The
#: SIGKILL has to land inside it, and the gap between observing the window and the signal
#: arriving is a syscall — so a second of window against a millisecond of latency is a
#: margin of roughly a thousand. Raising this buys more margin at a linear cost in runtime;
#: it is deliberately not tuned to the smallest value that happened to work on one laptop.
CRASH_PACE_SECONDS = 0.5

#: Everything below polls rather than sleeps, so these only bound the *failure* case and
#: can be generous. A loaded CI runner is slow, and a flaky chaos test is worse than none.
POLL_INTERVAL = 0.02
CRASH_WINDOW_TIMEOUT = 180.0
COMPLETION_TIMEOUT = 300.0
SUBPROCESS_TIMEOUT = 600.0

#: Last-resort CLI invocation if the console script is not on PATH; `llmrun.cli:main` is
#: the entry point the contract fixes, so this works even without an installed script.
_CLI_BOOTSTRAP = "import sys; from llmrun.cli import main; sys.exit(main())"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _example() -> ModuleType:
    """Import the example by path, so the test does not depend on ``PYTHONPATH``.

    Cached because importing it registers the pipeline, and registering the same name twice
    is (correctly) a ``ValueError``.
    """
    assert EXAMPLE_PATH.is_file(), f"the example pipeline is missing: {EXAMPLE_PATH}"
    spec = importlib.util.spec_from_file_location("llmrun_example_summarise_thread", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cli(*args: str) -> list[str]:
    """Build a command line for the ``llmrun`` CLI.

    Prefers the console script next to the running interpreter — the venv pytest is being
    run from — then anything on PATH, then the entry point directly. The point is that a
    developer running ``pytest`` outside ``make`` still gets a working chaos suite.
    """
    local = Path(sys.executable).with_name("llmrun")
    if local.is_file() and os.access(local, os.X_OK):
        return [str(local), *args]

    found = shutil.which("llmrun")
    if found:
        return [found, *args]

    return [sys.executable, "-c", _CLI_BOOTSTRAP, *args]


def _environment(database_url: str, *, pace_seconds: float) -> dict[str, str]:
    """Environment for a subprocess: same database, echo provider, no API key anywhere."""
    env = dict(os.environ)
    env["LLMRUN_DATABASE_URL"] = database_url
    env["LLMRUN_PROVIDER"] = "echo"
    env[_example().PACE_ENV] = str(pace_seconds)
    # Belt and braces: --import takes a path, but a dotted fallback should also resolve.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), existing]))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_cli(args: list[str], env: Mapping[str, str], *, what: str) -> str:
    """Run a short-lived CLI command to completion and return its stdout."""
    result = subprocess.run(
        _cli(*args),
        cwd=REPO_ROOT,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, (
        f"{what} failed with exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


def _enqueue(env: Mapping[str, str], db: Database, tenant: str) -> UUID:
    """Queue one run through the CLI and read its id back out of the database.

    Reading the id from ``runs`` rather than parsing the CLI's stdout keeps this test
    independent of how ``llmrun enqueue`` formats its output — that is the CLI's business,
    and not what is under test here.
    """
    _run_cli(
        [
            "enqueue",
            PIPELINE_NAME,
            "--tenant",
            tenant,
            "--json",
            json.dumps({"thread_id": THREAD_ID}),
            "--import",
            EXAMPLE_IMPORT_ARG,
        ],
        env,
        what="llmrun enqueue",
    )
    return db.latest_run_id(tenant)


@contextmanager
def _worker(env: Mapping[str, str], log_path: Path) -> Iterator[subprocess.Popen[str]]:
    """Start ``llmrun worker`` in its own process group, and guarantee it is gone after.

    ``start_new_session=True`` is what makes the group killable as a unit — signalling only
    the direct child would be a gentler experiment than the one this test claims to run.
    Output goes to a file rather than a pipe so a long-running worker can never block on a
    full pipe buffer while the test waits for it.
    """
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _cli(
            "worker",
            "--provider",
            "echo",
            "--import",
            EXAMPLE_IMPORT_ARG,
            "--lease-seconds",
            str(LEASE_SECONDS),
        ),
        cwd=REPO_ROOT,
        env=dict(env),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        yield process
    finally:
        _shutdown(process)
        handle.close()


def _shutdown(process: subprocess.Popen[str]) -> None:
    """Stop a worker politely, then rudely. Never leave one running after a test."""
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=30)


def _signal_group(process: subprocess.Popen[str], sig: int) -> None:
    with suppress(ProcessLookupError):  # already gone; nothing left to signal
        os.killpg(os.getpgid(process.pid), sig)


def _tail(log_path: Path, lines: int = 30) -> str:
    """The end of a worker's log, for assertion messages. Failures here are hard enough."""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - only on a disappearing tmp_path
        return f"<could not read {log_path}: {exc}>"
    return "\n".join(content.splitlines()[-lines:]) or "<no output>"


def _wait_for_crash_window(
    db: Database, run_id: UUID, minimum: int, *, log_path: Path, timeout: float
) -> tuple[int, int]:
    """Poll until the run is half done **and** a paid-for call is not yet checkpointed.

    Two conditions, and the second is what makes this experiment test the thing it claims
    to test.

    ``succeeded_steps >= minimum`` is the design contract's "poll until ~half the steps are
    succeeded". On its own it fires *between* steps, and a kill between steps is recovered
    entirely by the ``steps`` checkpoint table — the ``llm_calls`` ledger is never consulted,
    so the run would prove nothing about idempotency. The suite passed that way for a while,
    with zero ``cached=true`` rows to show for it.

    ``billed_calls >= succeeded_steps`` is the extra condition. Between steps every billed
    call has a committed step behind it and the calls run *behind* the step count (the fetch
    step costs nothing). The counts draw level only in the window where a provider call has
    been committed to the ledger and its step has not yet been checkpointed — which is
    precisely the crash the ledger exists to survive, and the only crash that forces the
    resumed worker to replay a call rather than a step.

    Polling, not sleeping: a fixed sleep either fires before the worker has started (and
    kills nothing interesting) or after it has finished (and proves nothing), depending on
    how loaded the machine is.
    """
    deadline = time.monotonic() + timeout
    succeeded = billed = 0
    while time.monotonic() < deadline:
        succeeded, billed = db.progress(run_id)
        if succeeded >= minimum and billed >= succeeded:
            return succeeded, billed
        time.sleep(POLL_INTERVAL)

    pytest.fail(
        f"never observed the crash window within {timeout:.0f}s: {succeeded} step(s) "
        f"succeeded and {billed} billed call(s) recorded, wanted at least {minimum} steps "
        f"with a call committed ahead of its checkpoint. Worker log tail:\n{_tail(log_path)}"
    )


def _wait_for_status(
    db: Database, run_id: UUID, status: str, *, log_path: Path, timeout: float
) -> RunRecord:
    """Poll ``runs`` until it reaches ``status``, or fail with everything worth knowing."""
    deadline = time.monotonic() + timeout
    record = db.run(run_id)
    while time.monotonic() < deadline:
        record = db.run(run_id)
        if record.status == status:
            return record
        if record.status in {"failed", "pending_review", "halted_budget"}:
            break
        time.sleep(POLL_INTERVAL)

    pytest.fail(
        f"run {run_id} is {record.status!r} (attempt {record.attempt}, "
        f"{db.count_steps(run_id, status='succeeded')} steps succeeded), expected "
        f"{status!r} within {timeout:.0f}s.\n"
        f"error_class={record.error_class!r} error_detail={record.error_detail!r}\n"
        f"Worker log tail:\n{_tail(log_path)}"
    )


# --------------------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------------------


def test_a_sigkilled_worker_resumes_without_paying_twice(
    db: Database,
    database_url: str,
    reset_tables: Callable[[], None],
    tmp_path: Path,
) -> None:
    example = _example()
    total_steps = len(example.STEP_NAMES)
    billed_steps = len(example.LLM_STEP_NAMES)
    quiet_env = _environment(database_url, pace_seconds=0)
    paced_env = _environment(database_url, pace_seconds=CRASH_PACE_SECONDS)

    # -- 1. the baseline: what this pipeline costs when nothing goes wrong ---------------
    baseline_run_id = _enqueue(quiet_env, db, BASELINE_TENANT)
    _run_cli(
        [
            "worker",
            "--once",
            "--provider",
            "echo",
            "--import",
            EXAMPLE_IMPORT_ARG,
            "--lease-seconds",
            str(LEASE_SECONDS),
        ],
        quiet_env,
        what="llmrun worker --once (baseline)",
    )

    baseline = db.run(baseline_run_id)
    assert baseline.status == "succeeded", (
        f"the baseline run did not finish cleanly: {baseline.status} "
        f"({baseline.error_class}: {baseline.error_detail})"
    )
    assert {step.name for step in db.steps(baseline_run_id)} == set(example.STEP_NAMES)
    baseline_output = baseline.output
    baseline_billed_calls = db.count_llm_calls(baseline_run_id, cached=False)
    assert baseline_billed_calls == billed_steps, (
        f"expected one billed call per LLM step ({billed_steps}), got {baseline_billed_calls}"
    )

    # -- 2. same work again, on an empty database ---------------------------------------
    reset_tables()
    crash_run_id = _enqueue(paced_env, db, CRASH_TENANT)
    assert crash_run_id != baseline_run_id

    # -- 3./4. run it for real, then kill the process group mid-flight ------------------
    crash_log = tmp_path / "worker-crashed.log"
    with _worker(paced_env, crash_log) as victim:
        succeeded_before_crash, billed_before_crash = _wait_for_crash_window(
            db,
            crash_run_id,
            total_steps // 2,
            log_path=crash_log,
            timeout=CRASH_WINDOW_TIMEOUT,
        )
        # SIGKILL, to the whole group. No signal handler runs, no transaction is flushed,
        # no lease is released — the worker simply stops existing between two statements.
        _signal_group(victim, signal.SIGKILL)
        returncode = victim.wait(timeout=60)

    assert returncode == -signal.SIGKILL, (
        f"the worker exited with {returncode} rather than being killed; "
        f"this run proved nothing. Log tail:\n{_tail(crash_log)}"
    )

    # If the run had already finished, every assertion below would pass vacuously — so the
    # experiment failing to interrupt anything has to be a test failure, not a quiet pass.
    interrupted = db.run(crash_run_id)
    assert interrupted.status != "succeeded", (
        "the pipeline finished before the SIGKILL landed, so nothing was interrupted and "
        f"this run proves nothing — raise CRASH_PACE_SECONDS (currently {CRASH_PACE_SECONDS})"
    )
    assert interrupted.status == "running", (
        f"expected the killed run to be stranded in 'running' with an expired lease, "
        f"found {interrupted.status!r}"
    )
    assert db.count_steps(crash_run_id, status="succeeded") < total_steps
    logger.info(
        "killed the worker with %d/%d steps committed and %d billed call(s) recorded",
        succeeded_before_crash,
        total_steps,
        billed_before_crash,
    )

    # The kill was aimed at the window where a call is paid for but its step is not yet
    # checkpointed. Confirm it landed there, because every assertion about the ledger below
    # is vacuous otherwise — a crash between steps is recovered by the `steps` table alone.
    stranded_steps, stranded_calls = db.progress(crash_run_id)
    assert stranded_calls > stranded_steps - 1, (
        f"the SIGKILL landed between steps ({stranded_calls} billed call(s), "
        f"{stranded_steps} checkpointed step(s)), so the crash never reached the window the "
        f"llm_calls ledger exists to cover. Log tail:\n{_tail(crash_log)}"
    )

    # -- 5. a fresh worker reclaims the expired lease and finishes the job ---------------
    resume_log = tmp_path / "worker-resumed.log"
    with _worker(quiet_env, resume_log) as survivor:
        final = _wait_for_status(
            db,
            crash_run_id,
            "succeeded",
            log_path=resume_log,
            timeout=COMPLETION_TIMEOUT,
        )
        assert survivor.poll() is None, (
            f"the second worker died before finishing:\n{_tail(resume_log)}"
        )

    # -- 6. the claims --------------------------------------------------------------------
    assert final.status == "succeeded"
    assert final.attempt >= 2, "the run was never actually reclaimed after the crash"

    steps = db.steps(crash_run_id)
    assert {step.name for step in steps} == set(example.STEP_NAMES)
    assert all(step.status == "succeeded" for step in steps)

    assert final.output == baseline_output, (
        "the resumed run produced different output from the uninterrupted one; resumption "
        "that changes the answer is not resumption"
    )

    crash_billed_calls = db.count_llm_calls(crash_run_id, cached=False)
    replayed_calls = db.count_llm_calls(crash_run_id, cached=True)

    # ==================================================================================
    #  THE NO-DOUBLE-BILLING ASSERTION.
    #
    #  A run that was killed halfway through and resumed by a different process must cost
    #  exactly what the uninterrupted run cost — not approximately, exactly. Every step
    #  that had committed before the kill is replayed from `steps`, and every provider
    #  call that had committed is replayed from the `llm_calls` ledger as a cached=true
    #  row worth $0.
    #
    #  This is the one assertion that fails the moment any of the following breaks:
    #    * input_hash stops being stable across processes (a resumed worker then misses
    #      the ledger and re-buys every completion it already owns),
    #    * a step's output is returned before its transaction commits,
    #    * the cache lookup is scoped wrongly, or written after the provider call instead
    #      of being consulted before it.
    #
    #  Each of those failures is invisible in every other way: the run still succeeds, the
    #  output is still correct, the logs still look healthy. The only symptom is the bill.
    #
    #  What this scenario does NOT reach: the worker below runs the example pipeline with a
    #  plain EchoProvider, so no step ever returns invalid JSON and `_complete`'s repair
    #  loop never runs twice. A crash landing between a *repair* and its checkpoint is a
    #  second, narrower window with the same failure mode, and it is covered in-process by
    #  test_pipeline.py::test_a_crash_between_a_repair_and_its_checkpoint_does_not_re_buy
    #  _the_repair rather than here, because forcing that interleaving from outside the
    #  process needs a kill aimed at a window a few milliseconds wide.
    # ==================================================================================
    assert crash_billed_calls == baseline_billed_calls, (
        f"CRASH RECOVERY DOUBLE-BILLED: the interrupted run made {crash_billed_calls} "
        f"billed provider call(s) but the clean baseline needed only "
        f"{baseline_billed_calls}. {replayed_calls} call(s) were replayed from the ledger. "
        f"Worker log tail:\n{_tail(resume_log)}"
    )

    # The companion to the assertion above, and the reason the kill is aimed at that
    # particular window. "Billed calls did not go up" is satisfied by two very different
    # systems: one that replays the interrupted call from the ledger, and one that has no
    # ledger at all but got lucky about where the crash landed. Only the first leaves a
    # `cached=true` row behind. Deleting the ledger lookup from context.py must fail this
    # test, not merely fail to be noticed by it.
    assert replayed_calls >= 1, (
        "no call was replayed from the ledger. The worker was killed with a provider call "
        "committed and its step not yet checkpointed, so the resumed run had to find that "
        "call in llm_calls — a re-purchase would have kept the totals equal only by "
        "coincidence."
    )

    # The ledger is append-only, so an absorbed retry shows up as an extra `cached=true`
    # row rather than as a mutated one: the total is the sum of the two, always.
    assert db.count_llm_calls(crash_run_id) == crash_billed_calls + replayed_calls
