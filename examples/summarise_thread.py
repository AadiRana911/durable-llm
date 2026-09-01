"""A twelve-step summarisation pipeline, written to be crashed on purpose.

Twelve checkpointed steps: one plain fetch, then eleven model calls that alternate between
*extracting* structure from the thread and *writing* prose from what was extracted. Nine of
those eleven hand :func:`ctx.llm` a ``schema`` callable — all but ``extract_risks`` and
``extract_open_questions``, whose output no later step parses — so most of the run is exposed
to the validate-then-repair path, where a malformed answer costs a bounded repair round inside
the step rather than the whole run.

It runs on the offline ``echo`` provider by default — no API key, no network, no cost —
and it is enqueueable straight from the CLI::

    llmrun enqueue summarise_thread --tenant demo \\
        --json '{"thread_id": "thread-42"}' --import examples/summarise_thread.py
    llmrun worker --once --provider echo --import examples/summarise_thread.py
    llmrun runs show <run-id>


What a mid-run crash costs
--------------------------

Assume the worker dies after step eight. Not a hypothetical: a rolling deploy, an OOM
kill, a reclaimed spot instance, a ``kubectl delete pod``. Seven billed model calls have
already been made and paid for.

**Without llmrun**, this pipeline is an ordinary Python function and its entire state —
``participants``, ``decisions``, ``actions`` — lives in local variables inside a process
that no longer exists. There is no checkpoint to resume from, so the only recovery is to
start again from the top. That means:

* the seven completed calls are **billed a second time**, and every subsequent retry of a
  run that keeps dying at step eight adds another seven;
* the user waits for the whole latency of the pipeline again, not the remaining third;
* nothing records that any of it happened, so the second invoice line is indistinguishable
  from legitimate traffic and the waste is invisible until someone reads the bill;
* and if the crash is a *refusal* at step eight rather than an infrastructure failure, the
  naive retry loop repeats those seven calls on every attempt while the eighth refuses
  every time — the failure mode that turns a monthly budget into an afternoon.

Eleven calls' worth of work becomes eighteen, then twenty-five, then thirty-two.

**With llmrun**, every step commits its output to Postgres before the next one starts, and
every real provider call is written to the ``llm_calls`` ledger in the same transaction as
the spend it caused. The restarted worker reclaims the run when its lease expires, replays
steps one through seven from their stored outputs *without calling the model*, and resumes
at step eight. The bill for the crash is exactly zero extra calls — which is not a claim
made in a README here, it is asserted in ``tests/chaos/test_crash_resume.py``:
``COUNT(*) FROM llm_calls WHERE cached = false`` after the crash must equal the count from
an uninterrupted run.


Notes on the shape of the code
------------------------------

*Step names are checkpoint keys.* A step's stored output is found by ``(run_id, name)``,
so renaming a step in a deploy is a cache invalidation: in-flight runs will re-execute it.
That is why the names live in :data:`STEP_NAMES` rather than being spelled inline twice.

*Every payload is plain JSON.* Steps and calls are content-addressed by hashing their
payload (see :mod:`llmrun.cache`), and only plain JSON types hash identically in a
different process. Sets, objects with a default ``__repr__``, and anything else whose
``str()`` embeds a memory address will hash differently after a restart and quietly cost
you a second call.

*Nothing here reads a clock or an RNG.* The synthetic thread is derived from a SHA-256 of
``thread_id`` so that two runs of the same input produce byte-identical output. The chaos
suite compares a crashed run's output against a clean baseline, and that comparison is
only meaningful if the pipeline is a pure function of its input.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Final

from llmrun.context import RunContext
from llmrun.pipeline import pipeline

logger = logging.getLogger("llmrun.examples.summarise_thread")

#: Output ceiling for every call. Also what the budget guard projects against before the
#: call is made, so it is a real cost lever and not just a truncation setting.
MAX_TOKENS: Final[int] = 512

#: Optional artificial delay per step, in seconds, read from the environment on every use.
#:
#: Zero by default, so ``make demo`` finishes instantly and CI is not slowed down. The
#: chaos suite sets it to a few hundred milliseconds for two reasons.
#:
#: The first is obvious: a SIGKILL has to land *mid-run* to prove anything, and on the echo
#: provider twelve steps otherwise complete faster than another process can observe them and
#: pull the trigger.
#:
#: The second is subtler and is why :func:`_pace` is also called from the schema validators.
#: There are two independent recovery mechanisms in llmrun, and a crash lands in the
#: territory of exactly one of them:
#:
#: * kill the worker *between* steps and the resumed run replays completed steps from the
#:   ``steps`` table — the ``llm_calls`` ledger is never consulted;
#: * kill it *after* a provider call has been committed to the ledger but *before* its step
#:   is checkpointed, and the resumed run has to find that call in ``llm_calls`` and replay
#:   it, or it buys the same completion twice.
#:
#: Only the second exercises the idempotency store. That window is the few milliseconds
#: between two commits inside ``ctx.llm()``, and pipeline code cannot reach into it — except
#: through the one callback llmrun invokes there: ``schema``. So the validators below pace
#: too, which widens the window to something a poller can reliably catch. A real pipeline
#: gets that width for free, from validation that does actual work.
PACE_ENV: Final[str] = "LLMRUN_DEMO_PACE_SECONDS"

#: The twelve steps, in order. Step one is a plain function call; the other eleven are
#: model calls. This tuple is the pipeline's contract with the chaos suite, which asserts
#: that a resumed run ends up with exactly these steps recorded as succeeded.
STEP_NAMES: Final[tuple[str, ...]] = (
    "fetch_messages",
    "extract_participants",
    "write_opening",
    "extract_decisions",
    "write_decisions_section",
    "extract_action_items",
    "write_action_items_section",
    "extract_risks",
    "write_risks_section",
    "extract_open_questions",
    "write_open_questions_section",
    "write_executive_summary",
)

#: The subset of :data:`STEP_NAMES` that costs money — every step except the fetch.
LLM_STEP_NAMES: Final[tuple[str, ...]] = STEP_NAMES[1:]

# --------------------------------------------------------------------------------------
# the synthetic corpus
# --------------------------------------------------------------------------------------

_SPEAKERS: Final[tuple[str, ...]] = (
    "ana.ruiz",
    "bo.chen",
    "priya.nair",
    "sam.okafor",
)

#: A fixed, plausible incident thread. Fixed on purpose: the point of the example is the
#: execution machinery, and a corpus that varies run to run would make the crash-resume
#: assertions meaningless.
_BODIES: Final[tuple[str, ...]] = (
    "Checkout latency crossed the 2s SLO at 09:14 UTC, mostly on the EU edge.",
    "Confirmed on our side — p99 is 4.1s and the error budget for the quarter is 60% gone.",
    "The regression lines up with the pricing-service deploy that went out at 09:02.",
    "Rolling that deploy back now. Give it five minutes to drain the old pods.",
    "Latency is back under 900ms. Error budget burn has stopped.",
    "Root cause looks like the new currency table is missing an index on (region, sku).",
    "Agreed. I'll write the migration, but it needs CONCURRENTLY — the table is 40M rows.",
    "Decision: we ship the index tonight, re-deploy pricing tomorrow behind a flag.",
    "Open question: do we backfill the cache warmers or let them repopulate naturally?",
    "Action: Priya writes the postmortem by Thursday, Sam owns the SLO dashboard fix.",
)


def fetch_messages(thread_id: str) -> dict[str, Any]:
    """The one non-model step: stand in for "go and read the ticket system".

    A real pipeline would query a database or an API here. What matters for the example is
    that this is *work worth checkpointing* — on resume, llmrun returns the stored output
    instead of hitting whatever system this represents a second time.

    Deterministic by construction: the speaker rotation is chosen from a SHA-256 of
    ``thread_id`` and the timestamps are computed from the message index, so no clock, no
    randomness, and the same ``thread_id`` always yields the same thread.
    """
    _pace()
    digest = hashlib.sha256(thread_id.encode("utf-8")).digest()
    offset = digest[0] % len(_SPEAKERS)

    messages = [
        {
            "index": index,
            "author": _SPEAKERS[(index + offset) % len(_SPEAKERS)],
            # Synthetic but well-formed timestamps: the database owns real time, not this.
            "sent_at": f"2026-03-{1 + index // 4:02d}T{9 + (index % 4) * 2:02d}:"
            f"{(7 * index) % 60:02d}:00+00:00",
            "body": body,
        }
        for index, body in enumerate(_BODIES)
    ]
    logger.debug("fetched %d synthetic messages for thread %s", len(messages), thread_id)
    return {
        "thread_id": thread_id,
        "message_count": len(messages),
        "messages": messages,
    }


# --------------------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------------------
#
# A `schema` is any callable that takes decoded JSON and either returns a validated value
# or raises. llmrun turns any exception raised here into a MalformedOutputError, which is
# retried *as a repair* — the validation error is appended to the prompt so the model can
# correct itself — and, if it still fails, ends the run in `pending_review` rather than
# `failed`, because a model that cannot satisfy a schema twice is a human's problem.
#
# Pydantic users would pass `Model.model_validate` here. These are plain functions to keep
# the example dependency-free, and to make the error messages the model sees legible: the
# text of the exception is what goes back into the repair prompt.


def validate_roster(parsed: Any) -> dict[str, Any]:
    """Validate the participant extraction and normalise it to a stable shape.

    Sorting the names is not cosmetic. This value is the payload of a later step, and step
    payloads are content-addressed — an unsorted list whose order depended on the model's
    mood would change the hash, miss the cache on resume, and buy the same completion
    twice.
    """
    _pace()
    body = _require_object(parsed)
    names = _require_string_list(body, "items")
    return {
        "participants": sorted(set(names)),
        "summary": _require_text(body, "summary"),
    }


def validate_findings(parsed: Any) -> dict[str, Any]:
    """Validate an extraction that carries a confidence score.

    ``confidence_pct`` stays an integer on purpose: this value is embedded in the run's
    JSON output, which the chaos suite compares byte-for-byte against a clean baseline, and
    integers survive a JSON round-trip exactly.
    """
    _pace()
    body = _require_object(parsed)
    score = body.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise ValueError(f"'score' must be an integer between 0 and 100, got {score!r}")
    return {
        "findings": _require_string_list(body, "items"),
        "summary": _require_text(body, "summary"),
        "confidence_pct": score,
    }


def validate_prose(parsed: Any) -> str:
    """Pull the prose out of the model's JSON envelope, and insist that there is some.

    Returning a plain ``str`` shows the other half of the ``schema`` contract: a validator
    is free to *transform*, not merely to check. The rest of the pipeline — and
    ``llmrun runs show`` — then deals in readable sentences instead of raw JSON.
    """
    _pace()
    return _require_text(_require_object(parsed), "summary")


def _require_object(parsed: Any) -> Mapping[str, Any]:
    if not isinstance(parsed, Mapping):
        raise TypeError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _require_text(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field!r} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_string_list(body: Mapping[str, Any], field: str) -> list[str]:
    value = body.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field!r} must be a non-empty list, got {value!r}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"every entry of {field!r} must be a non-empty string, got {value!r}")
    return [item.strip() for item in value]


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


@pipeline(name="summarise_thread", budget_usd="2.50", max_attempts=5)
def summarise_thread(ctx: RunContext, thread_id: str) -> dict[str, Any]:
    """Summarise one support thread in twelve checkpointed steps.

    Every ``ctx.step`` and every ``ctx.llm`` commits before the next one begins, so the
    twelve local variables below are not really local state — they are a projection of
    rows in Postgres. That is the entire trick: kill this process at any line and the
    replacement rebuilds every variable above the kill point from the database, for free,
    then carries on.

    The data flows forward: extractions read the thread, sections read the extraction that
    precedes them, and the executive summary reads the finished sections. A later step's
    payload therefore depends on an earlier step's output, which is exactly the property
    that makes a naive "just run it again" recovery so expensive.
    """
    thread = ctx.step("fetch_messages", fetch_messages, thread_id)

    participants = _extract(
        ctx,
        "extract_participants",
        "List every participant in this thread and summarise their role.",
        thread,
        schema=validate_roster,
    )
    opening = _write(
        ctx,
        "write_opening",
        "Write the opening paragraph of a summary for these participants.",
        {"thread_id": thread_id, "participants": participants["participants"]},
    )

    decisions = _extract(
        ctx,
        "extract_decisions",
        "Extract every decision the thread reached, with a confidence score.",
        thread,
        schema=validate_findings,
    )
    decisions_section = _write(
        ctx,
        "write_decisions_section",
        "Write the 'Decisions' section from these findings.",
        decisions,
    )

    actions = _extract(
        ctx,
        "extract_action_items",
        "Extract every action item, with an owner where one is named.",
        thread,
        schema=validate_findings,
    )
    actions_section = _write(
        ctx,
        "write_action_items_section",
        "Write the 'Action items' section, one line per item.",
        {"actions": actions, "participants": participants["participants"]},
    )

    # No schema on the two steps below: `ctx.llm` returns the raw completion text when a
    # schema is omitted. Prose that no downstream code parses does not need a validator,
    # and validating it anyway would only add a way for the run to fail.
    risks = _extract(
        ctx,
        "extract_risks",
        "Extract the residual risks that remain after these decisions.",
        thread,
    )
    risks_section = _write(
        ctx,
        "write_risks_section",
        "Write the 'Risks' section from this extraction.",
        {"risks": risks},
    )

    questions = _extract(
        ctx,
        "extract_open_questions",
        "Extract the questions the thread left unanswered.",
        thread,
    )
    questions_section = _write(
        ctx,
        "write_open_questions_section",
        "Write the 'Open questions' section from this extraction.",
        {"questions": questions},
    )

    executive_summary = _write(
        ctx,
        "write_executive_summary",
        "Write a three-sentence executive summary of the sections below.",
        {
            "opening": opening,
            "decisions": decisions_section,
            "action_items": actions_section,
            "risks": risks_section,
            "open_questions": questions_section,
        },
    )

    return {
        "thread_id": thread_id,
        "message_count": thread["message_count"],
        "participants": participants["participants"],
        "confidence_pct": {
            "decisions": decisions["confidence_pct"],
            "action_items": actions["confidence_pct"],
        },
        "sections": {
            "opening": opening,
            "decisions": decisions_section,
            "action_items": actions_section,
            "risks": risks_section,
            "open_questions": questions_section,
        },
        "executive_summary": executive_summary,
    }


def _extract(
    ctx: RunContext,
    name: str,
    instruction: str,
    payload: Any,
    *,
    schema: Any = None,
) -> Any:
    """One extraction call: pull structure out of ``payload``.

    ``model`` is deliberately not passed, so the run uses whatever default the worker's
    provider supplies. That is what keeps this example provider-agnostic: the same twelve
    steps run against the offline echo fake and against a real Anthropic model without a
    line changing.
    """
    _pace()
    return ctx.llm(name, _prompt(name, instruction), payload, schema=schema, max_tokens=MAX_TOKENS)


def _write(ctx: RunContext, name: str, instruction: str, payload: Any) -> str:
    """One writing call: turn an earlier step's structure into prose."""
    _pace()
    return ctx.llm(
        name,
        _prompt(name, instruction),
        payload,
        schema=validate_prose,
        max_tokens=MAX_TOKENS,
    )


def _prompt(step: str, instruction: str) -> str:
    """Build the instruction text for a step.

    The step name is part of the prompt on purpose. It gives provider-side logs something
    to correlate a request against, and it is how the offline echo provider tells one step
    from another when the framework does not pass the name through ``params``.
    """
    return (
        f"[{step}] {instruction}\n"
        "Reply with a single JSON object containing 'summary' (a sentence), "
        "'items' (a list of short strings) and 'score' (0-100)."
    )


def _pace() -> None:
    """Sleep for :data:`PACE_ENV` seconds, if it is set to something positive.

    A pipeline that sleeps is obviously artificial; this one does it so the chaos suite can
    reliably interrupt a run rather than racing a twelve-step echo pipeline that finishes
    in a couple of hundred milliseconds. Unset — which is the default everywhere except the
    chaos suite — this function does nothing at all.
    """
    raw = os.environ.get(PACE_ENV, "").strip()
    if not raw:
        return
    try:
        seconds = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", PACE_ENV, raw)
        return
    if seconds > 0:
        time.sleep(seconds)


def _main() -> int:
    """Enqueue one run from Python and print the commands that drive it.

    This is a script entry point rather than library code, which is why it is allowed to
    print. It exists to show the other half of the API: ``Pipeline.enqueue`` is the same
    call ``llmrun enqueue`` makes, so an application can queue work directly without
    shelling out to the CLI.
    """
    import sys

    from llmrun.db import connect

    thread_id = sys.argv[1] if len(sys.argv) > 1 else "thread-42"
    with connect() as conn:
        run_id = summarise_thread.enqueue(tenant_id="demo", conn=conn, thread_id=thread_id)

    print(f"queued run {run_id}")
    print(f"  llmrun worker --once --provider echo --import {__file__}")
    print(f"  llmrun runs show {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
