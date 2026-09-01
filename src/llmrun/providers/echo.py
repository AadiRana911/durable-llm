"""A deterministic, offline provider — no network, no API key, no clock.

This is not a test double bolted on at the end; it is the provider llmrun is *demoed*
on. ``make demo`` and the whole chaos suite run against it, which is what makes "clone
the repo and watch a crashed pipeline resume without double-billing" a ninety-second
experience rather than a signup flow.

Determinism is load-bearing, not a nicety. The chaos test SIGKILLs a worker mid-run,
restarts it, and asserts the final output byte-for-byte equals a clean baseline. That
assertion is only meaningful if the same request produces the same response in a
different process, on a different machine, on a different day. So every response here is
derived from ``sha256`` of the request — never from ``random``, ``uuid``, ``time``, or
``hash()`` (which is salted per-process by default and would silently break the test on
some runs and not others).

**Response shape.** By default ``complete()`` returns a JSON *object*, so steps that pass
a ``schema=`` validator succeed offline. It always carries the same seven keys:
``digest`` (first 16 hex chars of the request hash), ``echo`` (always true), ``items``
(two to four hyphenated word pairs), ``model``, ``score`` (0-100), ``step``, and
``summary`` (a four-word sentence). Keys are emitted sorted, so the serialised text is
stable too, not just the object.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from llmrun.types import LLMResult

__all__ = ["EchoProvider"]

logger = logging.getLogger("llmrun.providers.echo")

#: A closed vocabulary indexed by digest bytes. Fixed and ordered on purpose: adding or
#: reordering a word changes every generated response and would invalidate any recorded
#: baseline, so treat this tuple as part of the fake's observable contract.
_WORDS: tuple[str, ...] = (
    "amber",
    "anchor",
    "basalt",
    "beacon",
    "cedar",
    "cipher",
    "cobalt",
    "comet",
    "copper",
    "crest",
    "delta",
    "drift",
    "ember",
    "fathom",
    "ferry",
    "flint",
    "garnet",
    "harbor",
    "indigo",
    "iris",
    "jasper",
    "kettle",
    "lantern",
    "ledger",
    "marble",
    "meridian",
    "nimbus",
    "onyx",
    "orchid",
    "pilot",
    "quarry",
    "quartz",
)

#: Keys under which the framework may hand us the current step name. The design contract
#: fixes ``complete()``'s signature but is silent on how a step name reaches a provider,
#: so accept the plausible spellings rather than betting on one.
_STEP_KEYS: tuple[str, ...] = ("step", "step_name", "name")

#: Roughly four characters per token. Wrong in general, stable in particular — which is
#: all a fake needs, and it matches the estimate `budget.project_cost` uses.
_CHARS_PER_TOKEN = 4


def _stable(value: Any) -> str:
    """Render a value as text that is identical in every process.

    ``json.dumps(..., default=str)`` is tempting, but ``str()`` of an arbitrary object
    embeds its memory address (``<Foo object at 0x7f...>``), which would make the digest
    — and therefore the whole run — non-reproducible. Anything not JSON-serialisable
    collapses to its type name instead.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


class EchoProvider:
    """An offline provider whose output is a pure function of its input.

    Args:
        model: Model name reported when the caller does not supply one. Note that a
            model the caller *does* supply is echoed back verbatim, so the cost ledger
            prices what the pipeline asked for. Run the demo with ``default_model="echo"``
            (priced at zero in :mod:`llmrun.cost`) to keep it free; point a budget test
            at a real model name to make the same fake accrue spend.
        fail_at: ``{step_name: exception}``. The exception is raised the **first** time
            that step is seen by *this instance* and never again, which is exactly the
            shape a "fails once, then succeeds on retry" test needs.
        always_fail_at: ``{step_name: exception}``, raised on **every** call for that step
            — the same set-versus-mapping distinction ``malformed_steps`` draws between
            "always" and "for the first n attempts", expressed as a second mapping here
            because a bare set of names has nowhere to put the exception to raise.
            Without it the in-step transient retry budget (``DISPOSITIONS`` gives
            ``TransientError`` five calls inside one ``_billed_call``) could only ever be
            driven to a single failure, so the branch that gives up after exhausting it was
            unreachable from a test — the one path where llmrun stops retrying and lets the
            step fail, and the one nobody could prove worked. Steps listed here are checked
            before ``fail_at`` and never consume its one-shot entry, so a step named in both
            fails forever rather than failing twice and recovering. Note that exhausting the
            budget with a ``TransientError`` also pays the real backoff ladder between calls
            (~15s at the default one-second base), so a test that wants the give-up branch
            in a hurry should shrink the disposition rather than wait for it.
        latency_seconds: Sleep before responding. The chaos test needs a run long
            enough to interrupt; without this, a twelve-step pipeline finishes before
            ``SIGKILL`` can land.
        malformed_steps: Steps that should emit invalid JSON, to drive the repair loop.
            A ``set`` means "always malformed" (the run should end ``pending_review``).
            A mapping of ``{step: n}`` means "malformed for the first *n* attempts, then
            valid", which exercises a repair that actually *succeeds*.

    Attributes:
        call_counts: Per-step count of :meth:`complete` calls on this instance. Handy
            for asserting a resumed run did not re-issue calls it had already recorded.
    """

    name = "echo"

    def __init__(
        self,
        *,
        model: str = "echo",
        fail_at: Mapping[str, BaseException] | None = None,
        always_fail_at: Mapping[str, BaseException] | None = None,
        latency_seconds: float = 0.0,
        malformed_steps: Iterable[str] | Mapping[str, int] | None = None,
    ) -> None:
        self.model = model
        self.latency_seconds = float(latency_seconds)
        # Copied, not aliased: popping from the caller's dict would make the hook
        # single-use across every provider built from the same fixture.
        self._fail_at: dict[str, BaseException] = dict(fail_at or {})
        # Copied for symmetry rather than necessity — nothing mutates this one — but a
        # fixture that hands the same dict to two providers should not be able to have one
        # of them change what the other does.
        self._always_fail_at: dict[str, BaseException] = dict(always_fail_at or {})
        self._malformed: dict[str, int] = _normalise_malformed(malformed_steps)
        self.call_counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------------

    def complete(
        self,
        *,
        model: str = "",
        prompt: str = "",
        payload: str = "",
        max_tokens: int = 1024,
        **params: Any,
    ) -> LLMResult:
        """Return a response derived from ``sha256`` of the request.

        Order of operations is fixed so tests can reason about it: count the call, fire any
        ``always_fail_at`` or ``fail_at`` hook, sleep, then build the body. The count is
        incremented *before* the failure check, so a step that always fails still reports
        how many calls it absorbed — which is the whole point of driving the in-step retry
        budget to exhaustion. Failing before the sleep keeps failure tests fast.
        """
        # `ctx.llm()` accepts `payload: Any` while the provider protocol says `str`.
        # Coerce rather than crash, using the same canonical form as the digest so a
        # dict payload still hashes identically across processes.
        payload_text = payload if isinstance(payload, str) else _stable(payload)
        step = self._resolve_step(prompt, params)
        effective_model = model or self.model

        with self._lock:
            self.call_counts[step] += 1
            attempt = self.call_counts[step]
            # `always_fail_at` is consulted first and short-circuits, so a step named in
            # both hooks keeps its one-shot `fail_at` entry unpopped forever. The
            # alternative — pop first, then fall through — would leave the two hooks
            # fighting over which exception the first call raises, for no gain: a step that
            # always fails cannot also demonstrate recovery.
            failure = self._always_fail_at.get(step)
            if failure is None:
                failure = self._fail_at.pop(step, None)

        if failure is not None:
            logger.debug("echo: injecting %s at step %r", type(failure).__name__, step)
            raise failure

        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)

        digest = hashlib.sha256(
            self._seed(
                model=effective_model,
                prompt=prompt,
                payload=payload_text,
                max_tokens=max_tokens,
                step=step,
                params=params,
            ).encode("utf-8")
        ).digest()

        text = self._render(
            digest=digest,
            step=step,
            model=effective_model,
            malformed=attempt <= self._malformed.get(step, 0),
        )

        tokens_in = max(1, (len(prompt) + len(payload_text)) // _CHARS_PER_TOKEN)
        # A real provider cannot exceed max_tokens, so neither do we; otherwise a
        # budget projected from max_tokens could be undershot by the actual charge.
        tokens_out = max(1, min(max_tokens, len(text) // _CHARS_PER_TOKEN))

        logger.debug(
            "echo: step=%s model=%s tokens_in=%d tokens_out=%d",
            step,
            effective_model,
            tokens_in,
            tokens_out,
        )
        return LLMResult(
            text=text,
            model=effective_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    # -- internals ----------------------------------------------------------------

    def _resolve_step(self, prompt: str, params: Mapping[str, Any]) -> str:
        """Work out which pipeline step this call belongs to.

        The framework normally passes the name through ``params``. If it does not, fall
        back to looking for a *configured hook name* inside the prompt — the search is
        restricted to names the caller explicitly registered, so it can only ever fire
        a hook someone asked for.
        """
        for key in _STEP_KEYS:
            value = params.get(key)
            if isinstance(value, str) and value:
                return value

        hook_names = set(self._fail_at) | set(self._always_fail_at) | set(self._malformed)
        for candidate in sorted(hook_names):
            if candidate and candidate in prompt:
                return candidate

        return "-"

    def _seed(
        self,
        *,
        model: str,
        prompt: str,
        payload: str,
        max_tokens: int,
        step: str,
        params: Mapping[str, Any],
    ) -> str:
        """Build the canonical string that gets hashed.

        Field order is fixed and every field is length-delimited by a unit separator, so
        no combination of inputs can collide by running together.
        """
        parts = [
            f"step={step}",
            f"model={model}",
            f"max_tokens={max_tokens}",
            f"prompt={prompt}",
            f"payload={payload}",
        ]
        parts.extend(
            f"param:{key}={_stable(params[key])}" for key in sorted(params) if key not in _STEP_KEYS
        )
        return "\x1f".join(parts)

    def _render(self, *, digest: bytes, step: str, model: str, malformed: bool) -> str:
        """Turn digest bytes into the response text."""

        def word(index: int) -> str:
            return _WORDS[digest[index] % len(_WORDS)]

        item_count = 2 + digest[4] % 3
        body = {
            "digest": digest.hex()[:16],
            "echo": True,
            "items": [f"{word(5 + 2 * i)}-{word(6 + 2 * i)}" for i in range(item_count)],
            "model": model,
            "score": digest[12] * 100 // 255,
            "step": step,
            "summary": f"{word(0).capitalize()} {word(1)} {word(2)} {word(3)}.",
        }
        text = json.dumps(body, sort_keys=True, ensure_ascii=False)

        if not malformed:
            return text

        # Mimic the most common real failure: a chatty preamble wrapped around JSON
        # that was cut off mid-object. Unparseable, deterministic, and realistic enough
        # that the repair prompt has something plausible to react to.
        return f"Sure! Here is the JSON you asked for:\n{text[:-1]},"


def _normalise_malformed(spec: Iterable[str] | Mapping[str, int] | None) -> dict[str, int]:
    """Normalise ``malformed_steps`` to ``{step: attempts_to_corrupt}``.

    A bare iterable of names means "corrupt every attempt", expressed as a count no
    realistic repair loop will reach.
    """
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        return {str(step): int(count) for step, count in spec.items()}
    return {str(step): 1_000_000 for step in spec}
