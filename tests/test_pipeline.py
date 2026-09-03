"""Pipeline execution: resumption, repair, and the failure classes that must not retry.

Everything here is driven through the real worker — ``run_once`` claims a real row from a
real ``runs`` table and executes it against the offline echo provider. Nothing is mocked,
because the behaviour under test *is* the interaction between the Python function and the
rows it leaves behind.

The load-bearing trick is the counter in :data:`STEP_CALLS`. It is incremented inside the
body of each step function, so it counts *executions*, not calls. Resumption is not proved
by a run finishing twice with the same output — a pipeline that re-ran everything from
scratch would also do that, just at full price. It is proved by the counter refusing to go
up on the second attempt, which means the step function body never ran and the model was
never asked again.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest

from durable_llm import context as context_module
from durable_llm.context import RunContext
from durable_llm.db import transaction
from durable_llm.errors import DISPOSITIONS, ErrorClass, PolicyRefusalError, TransientError
from durable_llm.pipeline import pipeline
from durable_llm.providers.echo import EchoProvider
from durable_llm.worker import run_once

if TYPE_CHECKING:
    from conftest import Database

WORKER_ID = "test-pipeline"

#: Executions of each step function body, by step name. The whole file hangs off this.
STEP_CALLS: Counter[str] = Counter()

#: Step names that should raise a :class:`TransientError` the *first* time they execute.
#: Emptied as the failures fire, so the retry succeeds.
FLAKY_ONCE: set[str] = set()

#: When non-empty, the next response that *passes* validation kills the run instead of
#: being checkpointed. See :func:`validate_then_crash_once`.
CRASH_ON_VALID: set[str] = set()


@pytest.fixture(autouse=True)
def _reset_module_state() -> Iterator[None]:
    """Module-level counters are per-test state; leaking them between tests hides bugs."""
    STEP_CALLS.clear()
    FLAKY_ONCE.clear()
    CRASH_ON_VALID.clear()
    yield
    STEP_CALLS.clear()
    FLAKY_ONCE.clear()
    CRASH_ON_VALID.clear()


def _prompt(step: str) -> str:
    """Prompt text carrying the step name.

    The offline provider uses the name to decide which test hook (``fail_at``,
    ``malformed_steps``) applies, and a real provider's logs benefit from the same thing.
    """
    return f"[{step}] Summarise the payload as a single JSON object."


def _drive(db: Database, provider: EchoProvider) -> UUID | None:
    """Claim and execute one run, exactly as ``durable-llm worker --once`` does."""
    return run_once(db.conn, worker_id=WORKER_ID, provider=provider, lease_seconds=30)


def _executed(name: str, value: Any) -> Any:
    """Record that a step body actually ran, and fire a one-shot failure if armed."""
    STEP_CALLS[name] += 1
    if name in FLAKY_ONCE:
        FLAKY_ONCE.discard(name)
        raise TransientError(f"{name}: upstream timed out")
    return value


def collect(subject: str) -> dict[str, Any]:
    """The cheap, checkpointable step every pipeline below starts with."""
    return _executed("collect", {"subject": subject, "items": [f"{subject}-{n}" for n in range(3)]})


def score(headline: str) -> dict[str, Any]:
    """A step *after* an LLM call, so a failure here leaves earlier work checkpointed."""
    return _executed("score", {"length": len(headline)})


def require_items(parsed: Any) -> dict[str, Any]:
    """A ``schema`` callable: validates and normalises, raising on anything unusable."""
    if not isinstance(parsed, Mapping):
        raise TypeError(f"expected a JSON object, got {type(parsed).__name__}")
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"'items' must be a non-empty list, got {items!r}")
    return {"items": [str(item) for item in items]}


# --------------------------------------------------------------------------------------
# pipelines under test
# --------------------------------------------------------------------------------------


@pipeline(name="test_pipeline_resume", budget_usd=None, max_attempts=5)
def resume_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """step -> llm -> step, so a failure in the last step has two checkpoints behind it."""
    collected = ctx.step("collect", collect, subject)
    headline = ctx.llm("headline", _prompt("headline"), collected)
    scored = ctx.step("score", score, headline)
    return {"collected": collected, "headline": headline, "scored": scored}


@pipeline(name="test_pipeline_repair", budget_usd=None, max_attempts=5)
def repair_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """One validated call, to exercise the repair loop."""
    collected = ctx.step("collect", collect, subject)
    facts = ctx.llm("extract_facts", _prompt("extract_facts"), collected, schema=require_items)
    return {"facts": facts}


def validate_then_crash_once(parsed: Any) -> dict[str, Any]:
    """``require_items``, but the first answer that survives it kills the run.

    This is the in-process stand-in for the SIGKILL the chaos suite fires. It lands in the
    one window the ledger exists to cover: the provider has answered, the ``llm_calls`` row
    for that answer is committed, and the step has *not* been checkpointed yet.

    ``BaseException`` rather than ``Exception`` on purpose — ``_apply_schema`` maps every
    ``Exception`` onto ``MalformedOutputError``, which is the opposite of a crash, and
    ``run_once`` catches ``Exception`` to record a verdict. A ``KeyboardInterrupt`` walks
    out through both and leaves the run exactly as a killed worker would: ``running``, with
    its ledger rows durable and its checkpoint missing.
    """
    validated = require_items(parsed)
    if CRASH_ON_VALID:
        CRASH_ON_VALID.clear()
        raise KeyboardInterrupt("worker killed after the ledger row committed")
    return validated


@pipeline(name="test_pipeline_repair_crash", budget_usd=None, max_attempts=5)
def repair_crash_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A validated call whose repair is interrupted before it can be checkpointed."""
    collected = ctx.step("collect", collect, subject)
    facts = ctx.llm(
        "extract_facts",
        _prompt("extract_facts"),
        collected,
        schema=validate_then_crash_once,
    )
    return {"facts": facts}


@pipeline(name="test_pipeline_prompt", budget_usd=None, max_attempts=5)
def prompt_pipeline(ctx: RunContext, subject: str, instruction: str) -> dict[str, Any]:
    """The instruction is run input, so a resumed run can be given a different one."""
    collected = ctx.step("collect", collect, subject)
    return {"answer": ctx.llm("answer", instruction, collected)}


@pipeline(name="test_pipeline_escalate", budget_usd=None, max_attempts=5)
def escalate_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A pipeline that decides, itself, that a human should look at this run."""
    ctx.step("collect", collect, subject)
    ctx.escalate(f"{subject} needs a human")


@pipeline(name="test_pipeline_refusal", budget_usd=None, max_attempts=5)
def refusal_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A pipeline whose provider refuses to answer."""
    collected = ctx.step("collect", collect, subject)
    return {"verdict": ctx.llm("classify", _prompt("classify"), collected)}


@pipeline(name="test_pipeline_bug", budget_usd=None, max_attempts=5)
def buggy_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A plain programming error: no amount of retrying debugs a KeyError."""
    collected = ctx.step("collect", collect, subject)
    return {"boom": collected["a_key_that_does_not_exist"]}


def emit_nan(subject: str) -> dict[str, Any]:
    """A step body whose output ``json.dumps`` accepts and ``jsonb`` does not.

    The non-finite floats — ``nan``, ``inf``, ``-inf`` — are the values Python's encoder does
    not complain about: it emits the bare tokens ``NaN`` / ``Infinity`` / ``-Infinity``, none
    of which are JSON and none of which Postgres will parse. Everything else that cannot be
    checkpointed — a ``datetime``, a set, an arbitrary object — raises inside ``json.dumps``
    and was always caught. ``nan`` stands in for the family here, and the enqueue-side test
    below parametrises all three.
    """
    return _executed("emit_nan", {"subject": subject, "score": float("nan")})


@pipeline(name="test_pipeline_nan", budget_usd=None, max_attempts=5)
def nan_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A pipeline whose only step returns a value that cannot be stored."""
    return ctx.step("emit_nan", emit_nan, subject)


def emit_nul(subject: str) -> dict[str, Any]:
    """A step body whose output survives every guard and dies on the ``::jsonb`` cast.

    ``NaN`` is caught one layer up, by ``_to_json_text``'s ``allow_nan=False``, so it never
    reaches the checkpoint write. A NUL inside a string is the value that does: ``json.dumps``
    encodes it as the perfectly legal escape ``\\u0000``, ``json.loads`` reads it straight
    back, and Postgres then refuses it — ``jsonb`` stores strings as ``text`` and ``text``
    cannot hold a NUL, so the cast raises ``UntranslatableCharacter``.

    That makes it the only value in reach of these tests that fails *inside* ``_finish_step``
    rather than before it, which is the case that decides whether the step row is recorded or
    stranded. It is not a contrived one either: a NUL is what a truncated read, a C string
    crossing a boundary, or a scraped PDF hands you.
    """
    return _executed("emit_nul", {"subject": subject, "note": "a\u0000b"})


@pipeline(name="test_pipeline_nul", budget_usd=None, max_attempts=5)
def nul_pipeline(ctx: RunContext, subject: str) -> dict[str, Any]:
    """A pipeline whose only step returns a value the checkpoint write itself rejects."""
    return ctx.step("emit_nul", emit_nul, subject)


class NoWaitClock:
    """Stand-in for ``context.time`` that records backoff waits instead of taking them.

    The in-step transient budget is five calls spaced by a jittered doubling ladder from a
    one-second base — roughly fifteen seconds of real sleeping to prove one branch. Recording
    the waits rather than serving them keeps the suite fast *and* makes the pacing assertable,
    which a plain ``lambda *_: None`` patch would not: "it retried five times" and "it retried
    five times without pausing between them" are different claims, and only one of them is
    the documented behaviour.

    ``perf_counter`` is delegated rather than faked because ``_billed_call`` uses it to time
    the provider, and a frozen clock would write a zero into ``llm_calls.duration_ms``.
    """

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def perf_counter(self) -> float:
        return time.perf_counter()


# --------------------------------------------------------------------------------------
# resumption
# --------------------------------------------------------------------------------------


def test_resumption_returns_the_stored_output_without_re_executing(db: Database) -> None:
    """The second attempt rebuilds the run entirely from checkpoints.

    ``STEP_CALLS`` not moving is the proof. A pipeline that quietly re-ran everything would
    produce the same output and the same ``succeeded`` status — and a second invoice.
    """
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    first_provider = EchoProvider()
    assert _drive(db, first_provider) == run_id
    first = db.run(run_id)
    assert first.status == "succeeded"
    assert STEP_CALLS["collect"] == 1
    assert STEP_CALLS["score"] == 1
    assert sum(first_provider.call_counts.values()) == 1

    # Exactly what a reclaimed lease does to a run, minus waiting for it to expire. The
    # stored output is cleared, so the second attempt cannot pass by accident.
    db.requeue(run_id)

    second_provider = EchoProvider()
    assert _drive(db, second_provider) == run_id

    second = db.run(run_id)
    assert second.status == "succeeded"
    assert second.output == first.output
    assert STEP_CALLS["collect"] == 1, "the step body re-executed instead of being replayed"
    assert STEP_CALLS["score"] == 1, "the step body re-executed instead of being replayed"
    assert sum(second_provider.call_counts.values()) == 0, "the model was asked again"
    assert db.count_llm_calls(run_id, cached=False) == 1, "a replay was billed as a new call"


def test_changed_input_re_executes_the_step(db: Database) -> None:
    """Resumption is keyed on the input hash, not on the step name.

    Serving a stored output to a step whose arguments have changed would be a correctness
    bug dressed up as a cache — quietly returning last week's answer.
    """
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()
    assert _drive(db, EchoProvider()) == run_id
    first = db.run(run_id)

    db.requeue(run_id, run_input={"subject": "beta"})
    assert _drive(db, EchoProvider()) == run_id

    second = db.run(run_id)
    assert second.status == "succeeded"
    assert STEP_CALLS["collect"] == 2, "changed arguments must re-execute the step"
    assert second.output != first.output
    assert second.output["collected"]["subject"] == "beta"
    # A different payload is a different call, so it is billed — correctly — as a new one.
    assert db.count_llm_calls(run_id, cached=False) == 2


def test_a_successful_run_checkpoints_every_step(db: Database) -> None:
    """Both ``ctx.step`` and ``ctx.llm`` leave a ``steps`` row behind.

    The chaos suite watches that table to decide when to pull the trigger, so "an LLM call
    is also a checkpoint" is load-bearing well beyond this file.
    """
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, EchoProvider()) == run_id

    steps = db.steps(run_id)
    assert {step.name for step in steps} == {"collect", "headline", "score"}
    assert all(step.status == "succeeded" for step in steps)
    assert all(step.ended_at is not None for step in steps)
    assert all(step.input_hash for step in steps)
    assert db.count_llm_calls(run_id, cached=False) == 1


# --------------------------------------------------------------------------------------
# transient failures
# --------------------------------------------------------------------------------------


def test_a_transient_step_failure_is_retried_and_earlier_work_is_replayed(
    db: Database,
) -> None:
    """The crash-resume story in miniature, without the SIGKILL.

    The third step fails once. The worker requeues the run, and the second attempt replays
    the first two steps from Postgres — the collect body does not run again and the model
    is not called again — then executes only the step that actually failed.
    """
    FLAKY_ONCE.add("score")
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    first_provider = EchoProvider()
    assert _drive(db, first_provider) == run_id
    after_failure = db.run(run_id)
    assert after_failure.status == "queued", "a transient failure must go back on the queue"
    assert sum(first_provider.call_counts.values()) == 1

    # A requeued run is paced by `runs.next_attempt_at` (migrations/0003) so a failing run
    # is not re-claimed on the next poll. This test is about what the *second* attempt
    # replays, not about how long it waits, so the wait is stepped over rather than slept
    # through — test_worker.py covers the pacing itself.
    db.clear_backoff(run_id)

    second_provider = EchoProvider()
    assert _drive(db, second_provider) == run_id

    final = db.run(run_id)
    assert final.status == "succeeded"
    assert final.attempt == 2
    assert STEP_CALLS["collect"] == 1, "a succeeded step must not re-execute after a retry"
    assert STEP_CALLS["score"] == 2, "the failed step must re-execute"
    assert sum(second_provider.call_counts.values()) == 0
    assert db.count_llm_calls(run_id, cached=False) == 1, "the retry re-billed a completed call"


def test_a_transient_provider_error_is_retried_inside_the_step(db: Database) -> None:
    """A 429 does not need a whole new run — ``ctx.llm`` retries in place, with backoff.

    A failed call is never billed, so no ledger row is written for the first attempt.
    """
    provider = EchoProvider(fail_at={"headline": TransientError("429 from upstream")})
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, provider) == run_id

    record = db.run(run_id)
    assert record.status == "succeeded"
    assert record.attempt == 1, "the run itself should not have needed a second attempt"
    assert sum(provider.call_counts.values()) == 2, "the transient failure was not retried"
    assert db.count_llm_calls(run_id, cached=False) == 1


# --------------------------------------------------------------------------------------
# malformed output and the repair loop
# --------------------------------------------------------------------------------------


def test_malformed_output_is_repaired_on_the_second_attempt(db: Database) -> None:
    """The model returns unparseable JSON once, then valid JSON, and the run succeeds.

    A *plain* retry would not fix this: the same prompt tends to produce the same
    malformed answer, and you pay for both. The repair attempt feeds the validation error
    back into the prompt, which is also why the second attempt must not be served from the
    first attempt's ledger entry — if the repair hashes identically to the failed call, the
    cache replays the malformed response and the loop can never converge.
    """
    provider = EchoProvider(malformed_steps={"extract_facts": 1})
    run_id = repair_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, provider) == run_id

    record = db.run(run_id)
    assert record.status == "succeeded"
    assert record.attempt == 1, "the repair belongs inside the step, not to a new run"
    assert sum(provider.call_counts.values()) == 2, "the malformed answer was not repaired"
    assert record.output["facts"]["items"], "the validated object should have reached the output"


def test_output_that_never_validates_ends_in_pending_review(db: Database) -> None:
    """Two failed repairs is a human's problem, not a ``failed`` run.

    ``pending_review`` is a different status on purpose: nothing is broken, the model
    simply cannot produce what the schema demands, and retrying it forever only spends.
    """
    provider = EchoProvider(malformed_steps=["extract_facts"])
    run_id = repair_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, provider) == run_id

    record = db.run(run_id)
    assert record.status == "pending_review"
    assert record.error_class == "malformed"
    assert _drive(db, EchoProvider()) is None, "a run awaiting review must not be re-claimed"

    # DISPOSITIONS[MALFORMED].max_attempts is 2, and errors.Disposition documents
    # max_attempts as counting *total* attempts rather than extra ones — which is also how
    # context._billed_call reads it for TRANSIENT (max_calls = max_attempts). The repair
    # loop must read the same field the same way, or one number in the disposition table
    # means two different things depending on which loop is looking at it.
    calls = sum(provider.call_counts.values())
    assert calls == DISPOSITIONS[ErrorClass.MALFORMED].max_attempts, (
        f"the repair loop made {calls} provider call(s) for a max_attempts of "
        f"{DISPOSITIONS[ErrorClass.MALFORMED].max_attempts}; every one of them was billed"
    )


def test_a_crash_between_a_repair_and_its_checkpoint_does_not_re_buy_the_repair(
    db: Database,
) -> None:
    """The intersection the other two repair tests each miss half of.

    ``test_malformed_output_is_repaired_on_the_second_attempt`` covers a repair with no
    crash; ``test_resumption_returns_the_stored_output_without_re_executing`` covers a crash
    with no repair. A step that needed a repair *and then* was killed before its checkpoint
    committed is the case where the money actually leaks, and it leaks silently: the output
    is still correct, so only the ledger shows it.

    Both answers are already paid for and both are in ``llm_calls``, so the resumed step
    must replay *both* — the malformed one to fail validation exactly as it did the first
    time, and the repair to satisfy it. A resume that re-buys the repair does so on every
    resume, forever, which is what makes this worth a test of its own.
    """
    provider = EchoProvider(malformed_steps={"extract_facts": 1})
    run_id = repair_crash_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    CRASH_ON_VALID.add("extract_facts")
    with pytest.raises(KeyboardInterrupt):
        _drive(db, provider)

    # One malformed answer plus one repair, both billed, neither checkpointed.
    assert sum(provider.call_counts.values()) == 2
    assert db.count_llm_calls(run_id, cached=False) == 2
    assert db.count_steps(run_id, status="succeeded") == 1, "only `collect` should have landed"

    db.requeue(run_id)
    resumed_provider = EchoProvider(malformed_steps={"extract_facts": 1})
    assert _drive(db, resumed_provider) == run_id

    record = db.run(run_id)
    assert record.status == "succeeded"
    assert record.output["facts"]["items"], "the repaired answer should have reached the output"
    assert sum(resumed_provider.call_counts.values()) == 0, (
        "the resumed step called the provider again; every answer it needed was already "
        "paid for and recorded in llm_calls"
    )
    assert db.count_llm_calls(run_id, cached=False) == 2, (
        "the resume re-bought an answer it already had — this is the double-billing the "
        "ledger exists to prevent, on the one path that reaches it twice"
    )
    assert db.count_llm_calls(run_id, cached=True) == 2, (
        "both the malformed answer and its repair should be recorded as absorbed replays"
    )


def test_editing_a_prompt_re_executes_the_step_instead_of_replaying_the_old_answer(
    db: Database,
) -> None:
    """The prompt is part of the cache key, because it is the input that decides the answer.

    Editing an instruction and restarting an in-flight run is the single most common
    pipeline edit, and the one a deploy is most likely to ship mid-run. If the prompt is not
    hashed, the resumed run serves the answer the *old* instruction produced — with no
    error, no log line and no ledger evidence that it did.
    """
    run_id = prompt_pipeline.enqueue(
        tenant_id="pipeline",
        conn=db.conn,
        subject="alpha",
        instruction="Summarise this thread in one sentence.",
    )
    db.conn.commit()
    assert _drive(db, EchoProvider()) == run_id
    first = db.run(run_id)

    db.requeue(
        run_id,
        run_input={
            "subject": "alpha",
            "instruction": "TRANSLATE this thread into French. Do not summarise.",
        },
    )
    provider = EchoProvider()
    assert _drive(db, provider) == run_id

    second = db.run(run_id)
    assert second.status == "succeeded"
    assert STEP_CALLS["collect"] == 1, "the unchanged step should still replay"
    assert sum(provider.call_counts.values()) == 1, (
        "the changed instruction was served from the checkpoint instead of being asked"
    )
    assert second.output["answer"] != first.output["answer"]
    assert db.count_llm_calls(run_id, cached=False) == 2

    # And with the checkpoint gone, the ledger must not serve the old answer either.
    db.requeue(run_id)
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM steps WHERE run_id = %s::uuid AND name = 'answer'", (str(run_id),))
    db.conn.commit()

    third_provider = EchoProvider()
    assert _drive(db, third_provider) == run_id
    assert db.run(run_id).output["answer"] == second.output["answer"], (
        "the ledger replayed the answer to a prompt that is no longer being asked"
    )
    assert sum(third_provider.call_counts.values()) == 0


# --------------------------------------------------------------------------------------
# policy refusals — the class that must never be retried
# --------------------------------------------------------------------------------------


def test_a_policy_refusal_ends_the_run_pending_review_and_is_not_retried(db: Database) -> None:
    """``ctx.escalate`` routes a run to a human instead of failing it.

    Retrying a refusal is the purest way to burn money: every attempt is billed, every
    attempt is refused, and the loop stops only when the budget does.
    """
    run_id = escalate_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, EchoProvider()) == run_id

    record = db.run(run_id)
    assert record.status == "pending_review"
    assert record.error_class == "policy"
    assert "needs a human" in (record.error_detail or "")
    assert record.attempt == 1

    assert _drive(db, EchoProvider()) is None, "a refusal must not be retried"
    assert db.run(run_id).status == "pending_review"


def test_a_provider_refusal_is_billed_once_and_never_repeated(db: Database) -> None:
    """The provider refuses; ``ctx.llm`` must not treat that as a transient blip."""
    provider = EchoProvider(fail_at={"classify": PolicyRefusalError("I can't help with that")})
    run_id = refusal_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, provider) == run_id

    record = db.run(run_id)
    assert record.status == "pending_review"
    assert record.error_class == "policy"
    assert sum(provider.call_counts.values()) == 1, "a refusal was retried, and retries are billed"
    assert _drive(db, EchoProvider()) is None


# --------------------------------------------------------------------------------------
# permanent failures
# --------------------------------------------------------------------------------------


def test_a_bug_in_pipeline_code_fails_fast(db: Database) -> None:
    """An unclassified exception is assumed to be a bug, and bugs are not retried.

    The alternative — defaulting to transient — turns one stack trace into five, spaced out
    by exponential backoff, each one having paid for whatever ran before the crash.
    """
    run_id = buggy_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, EchoProvider()) == run_id

    record = db.run(run_id)
    assert record.status == "failed"
    assert record.error_class == "permanent"
    assert record.attempt == 1
    assert STEP_CALLS["collect"] == 1
    assert _drive(db, EchoProvider()) is None, "a permanent failure must not be retried"


def test_the_in_step_transient_budget_is_bounded_never_billed_and_paced(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch where durable_llm stops retrying a provider and lets the step fail.

    ``ctx.llm`` retries a transient failure in place, and the disposition table says how many
    times. Nothing tested the *end* of that budget, because the echo provider's ``fail_at``
    hook is consumed by the first call — so every transient test drove exactly one failure
    and the give-up branch was unreachable. ``always_fail_at`` makes the step fail forever,
    which is what a genuinely down provider does.

    Three things have to hold at the end of it, and each is a different way of losing money
    or a run:

    * the provider is called exactly ``DISPOSITIONS[TRANSIENT].max_attempts`` times. One more
      and the table means nothing; one fewer and a blip that would have cleared on the last
      attempt costs a whole extra run-level retry;
    * **no billed ledger row is written at all.** A call that raised was never answered and
      is never charged, so a ledger row for one would inflate ``spent_usd``, eat the budget
      cap, and — because the row is keyed by ``(run_id, input_hash)`` — hand a resumed run a
      replay with no response text in it;
    * the run goes back on the queue rather than terminal. The step gave up; the *run* has
      not, and a provider that is down for thirty seconds must not be a permanent failure.

    The pacing is asserted too, from the recorded waits: retrying a failing provider five
    times as fast as the loop can issue them is the thundering herd the backoff exists to
    prevent, and it would still satisfy the call count above.
    """
    clock = NoWaitClock()
    monkeypatch.setattr(context_module, "time", clock)

    budget = DISPOSITIONS[ErrorClass.TRANSIENT]
    provider = EchoProvider(always_fail_at={"headline": TransientError("429 from upstream")})
    run_id = resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, provider) == run_id

    calls = provider.call_counts["headline"]
    assert calls == budget.max_attempts, (
        f"the step made {calls} provider call(s) against a transient budget of "
        f"{budget.max_attempts}; the in-step retry loop and the disposition table disagree"
    )
    assert db.count_llm_calls(run_id) == 0, (
        "a failed provider call left a row in llm_calls; a call that raised was never "
        "answered and must never be billed"
    )
    assert len(clock.slept) == budget.max_attempts - 1, (
        "the retries were not spaced — a failing provider was answered with every remaining "
        "attempt as fast as the loop could issue them"
    )
    assert all(delay > 0 for delay in clock.slept)

    record = db.run(run_id)
    assert record.status == "queued", "an exhausted in-step budget must not end the run"
    assert record.error_class == "transient"
    assert record.failures == 1, "the whole in-step budget is one run-level failure, not five"

    steps = {step.name: step for step in db.steps(run_id)}
    assert steps["collect"].status == "succeeded", "work done before the outage was thrown away"
    assert steps["headline"].status == "failed"
    assert STEP_CALLS["collect"] == 1


# --------------------------------------------------------------------------------------
# values that cannot be stored — the guards that keep a psycopg error from surfacing
# --------------------------------------------------------------------------------------


def test_a_step_returning_nan_fails_the_run_by_name_rather_than_by_psycopg(db: Database) -> None:
    """``NaN`` is the value that used to get past the checkpoint guard.

    ``_to_json_text`` exists to turn "this step returned something uncheckpointable" into a
    ``PermanentError`` that names the step. It missed ``allow_nan=False``, and Python's
    encoder emits the bare token ``NaN`` rather than raising — so the value serialised
    happily, round-tripped through ``json.loads`` happily, and died inside the checkpoint
    write as a raw ``psycopg.errors.InvalidTextRepresentation`` about SQL syntax. The author
    of a twelve-step pipeline then had a failed run whose ``error_detail`` did not say which
    step, or that the problem was a float.

    The step row matters as much as the message. ``_to_json_text`` is called inside
    ``_checkpointed``'s ``try``, so the ``PermanentError`` it raises reaches ``_fail_step``
    and the step is recorded as ``failed`` rather than left at ``running`` — the state that
    reads, forever, as work still in progress. Delete the ``self._fail_step(name, exc)`` line
    and this test goes red on that assertion alone.

    Note what this test does *not* cover: with the guard in place, ``NaN`` never reaches
    ``_finish_step`` at all, so it says nothing about whether the *checkpoint write* is
    inside the ``try``. That needs a value the encoder accepts and Postgres rejects — see
    :func:`test_a_failing_checkpoint_write_records_the_step_rather_than_stranding_it`.
    """
    run_id = nan_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, EchoProvider()) == run_id

    record = db.run(run_id)
    assert record.status == "failed"
    assert record.error_class == "permanent"
    detail = record.error_detail or ""
    assert detail.startswith("PermanentError:"), (
        f"the run failed with a raw driver error instead of a diagnosis: {detail!r}"
    )
    assert "'emit_nan'" in detail, "the failure does not name the step that produced the value"
    assert "cannot be checkpointed" in detail
    assert "InvalidTextRepresentation" not in detail

    steps = db.steps(run_id)
    assert len(steps) == 1
    assert steps[0].status == "failed", (
        "the step that could not be checkpointed was left at 'running', which is "
        "indistinguishable from a step still executing"
    )
    assert steps[0].error_class == "permanent"
    assert steps[0].output is None


def test_a_failing_checkpoint_write_records_the_step_rather_than_stranding_it(
    db: Database,
) -> None:
    """``_finish_step`` is inside ``_checkpointed``'s ``try``, and this is why.

    Every other "unstorable value" test is really a test of a guard that fires *before* the
    write: ``json.dumps`` raises, or ``allow_nan=False`` raises, and the checkpoint statement
    is never issued. That leaves the write itself untested, and the write is the part that
    can fail for reasons no guard can anticipate — the lease moved on, the connection died,
    or the value survived ``json.dumps`` and not the ``::jsonb`` cast.

    A NUL inside a string is the last of those, reachable from ordinary pipeline code (see
    :func:`emit_nul`). It gets all the way into ``_finish_step``'s ``UPDATE`` and dies there.
    What happens next is the property: the step must end up recorded as ``failed`` with no
    output, not stranded at ``running``. ``running`` is the single worst state a step can be
    left in — the resumed run re-executes it (correct but silent), ``durable-llm runs show``
    reports work in progress on a run nothing is working on, and nothing anywhere says why.

    Deliberately *not* asserted: that ``error_detail`` is a friendly diagnosis. It is a raw
    driver message here, because no guard claims to catch this class of failure — the claim
    under test is only that the failure is written down instead of being lost.
    """
    run_id = nul_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject="alpha")
    db.conn.commit()

    assert _drive(db, EchoProvider()) == run_id
    assert STEP_CALLS["emit_nul"] == 1, "precondition: the step body ran and produced a value"

    record = db.run(run_id)
    assert record.status == "failed"
    assert record.error_class == "permanent", "a value jsonb cannot store is not worth retrying"

    steps = db.steps(run_id)
    assert len(steps) == 1
    assert steps[0].name == "emit_nul"
    assert steps[0].status == "failed", (
        "the checkpoint write failed and left the step at 'running': indistinguishable from "
        "a step still executing, on a run no worker holds"
    )
    assert steps[0].output is None, "a step whose write failed must not appear to have stored one"
    assert steps[0].error_detail, "the step failed with no record of why"


@pytest.mark.parametrize(
    ("value", "description"),
    [
        (float("nan"), "a bare NaN"),
        (float("inf"), "a bare Infinity"),
        (float("-inf"), "a bare -Infinity"),
        ({"a": [float("inf")]}, "an Infinity nested inside a list inside a dict"),
        ([{"score": float("nan")}], "a NaN nested inside a dict inside a list"),
    ],
    ids=["nan", "inf", "negative-inf", "nested-inf", "nested-nan"],
)
def test_enqueueing_a_non_finite_argument_is_refused_at_the_call_site(
    db: Database,
    value: Any,
    description: str,
) -> None:
    """``enqueue`` promises "a TypeError here rather than a failed run later".

    Every other unstorable argument keeps that promise for free, because ``json.dumps``
    raises on it. The non-finite floats are the exception: the encoder emits ``NaN`` /
    ``Infinity`` / ``-Infinity``, which are not JSON and which ``jsonb`` will not parse, so
    the argument sailed past the guard and died on the INSERT as a
    ``psycopg.errors.InvalidTextRepresentation`` — a message about SQL text, raised from
    inside a library, describing an argument the caller passed several frames up.

    Nesting is parametrised because the guard has to be the encoder rather than a check on
    the top-level values: an ``inf`` three levels down is exactly as unstorable and far more
    likely to arrive from real code than a bare one.

    The row count is the other half. A guard that raised *after* the insert would leave a run
    queued against arguments the worker cannot read back.
    """
    del description  # named for the test id and for anyone reading the parameter list

    with pytest.raises(TypeError, match="JSON-serialisable"):
        resume_pipeline.enqueue(tenant_id="pipeline", conn=db.conn, subject=value)

    with transaction(db.conn) as cur:
        cur.execute("SELECT count(*) AS n FROM runs")
        row = cur.fetchone()
    assert row is not None
    assert int(row["n"]) == 0, "the rejected enqueue still wrote a row"
