"""Content addressing for steps and LLM calls.

Everything llmrun claims about not double-billing rests on one property:

    **The same logical input must produce the same hash — in a different process, on a
    different machine, on a different Python version, a week later.**

The idempotency store is a Postgres table keyed by ``(run_id, input_hash)``. When a worker
is SIGKILLed after the provider responded but before the transaction committed, the
restarted worker recomputes the hash, finds the earlier ``llm_calls`` row, and replays the
stored response instead of paying for it again. A hash that varies by even one bit between
those two processes turns a replay into a second billed call — and it fails *silently*,
because a cache miss looks exactly like normal operation. That is why the canonicalisation
rules below are pinned rather than left to whatever ``json.dumps`` defaults to.

The rules, and what each buys:

``sort_keys=True``
    Mapping iteration order stops mattering, so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    are the same input. Without it, a payload assembled by a different code path — or by
    the same code path after a refactor — hashes differently.

``separators=(",", ":")``
    No insignificant whitespace, so pretty-printing or reformatting upstream cannot change
    the key.

``ensure_ascii=False`` plus an explicit UTF-8 encode
    One canonical byte representation of non-ASCII text, instead of depending on the
    platform's default encoding.

A type-tagged ``default`` fallback
    Values JSON cannot represent natively (``Decimal``, ``UUID``, ``datetime``, enums) are
    reduced to ``{"__type__": type(o).__name__, "__str__": str(o)}`` rather than raising, so
    hashing never becomes the reason a pipeline crashes. The tag is the load-bearing half.
    A plain ``default=str`` — which this used to be — is type-*blind*: ``Decimal("1")``,
    the string ``"1"`` and a ``UUID`` all collapse to the same characters, so two genuinely
    different calls produce the same key and the second one is served the first one's stored
    response. That is not a missed cache, it is a **wrong answer**, returned confidently,
    with a ledger row that says it was a legitimate replay. Carrying the type name alongside
    the text keeps those inputs distinct while leaving the crash-proofing intact.

Two hazards remain, both worth knowing before you put exotic objects in a payload. Note
that neither is fixed by the tag — the tag makes types distinguishable, not ``str()``
deterministic:

* **Sets and frozensets are not safe.** They reach ``str()`` through the tag, and its
  output depends on iteration order, which for strings depends on ``PYTHONHASHSEED`` and
  therefore differs between processes. Convert them to a sorted list first.
* **Objects with a default ``__repr__`` are not safe** — ``<Foo object at 0x7f...>``
  embeds a memory address. Give payload objects a stable ``__str__``, or reduce them to
  primitives before hashing.

Both produce *unstable* hashes, which cost money rather than corrupting anything: a stale
key simply misses the cache and re-bills. That is the mild failure; the collision described
above was the severe one, and it is the reason the fallback is no longer ``str``. Prefer
plain JSON types in payloads and neither can reach you.

Adopting the tag changed the bytes fed to the hash, so — exactly as with
:data:`HASH_ALGORITHM` — every key computed by an older version is now unreachable. A
database carrying cached calls from before this change will miss once per distinct call and
re-bill it, after which the new keys are in the ledger and replay resumes. Nothing is
corrupted and no migration is needed; it is a one-off cost paid to stop serving one call's
answer to another call's input.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["HASH_ALGORITHM", "canonical_json", "input_hash", "prompt_json"]

#: Named so it can be asserted in tests and referenced in a future migration. Changing it
#: invalidates every cached call in existing runs, so it is not a casual edit.
HASH_ALGORITHM: Final[str] = "sha256"


def _tagged(obj: Any) -> dict[str, str]:
    """Reduce a non-JSON value to its type name plus its text, for :func:`canonical_json`.

    Two properties matter and they pull in opposite directions. Hashing must not raise —
    an exotic type in a payload is a nuisance, not a reason to fail a run — which argues
    for the broadest possible fallback. But the fallback must not merge distinct values,
    because the hash is an idempotency key: anything that collapses two different inputs
    to the same characters makes the ledger hand the second call the first call's stored
    response. ``str()`` alone merged ``Decimal("1")``, ``"1"`` and a ``UUID`` whose text
    happens to match; prefixing the type name separates them at no cost to the first
    property. ``type(o).__name__`` rather than the fully qualified name because the module
    path is refactorable and the key must survive a module move.

    One collision does survive, and it is worth naming rather than quietly leaving for
    someone to rediscover: a payload containing the literal dict ``{"__type__": "Decimal",
    "__str__": "1"}`` hashes exactly as ``Decimal("1")`` does, because the tag is ordinary
    data and JSON has nowhere else to put it. Closing that would mean escaping every dict in
    every payload — invalidating every key a second time, for everyone, to defend against a
    shape no one writes by accident. The tag is aimed at the collision that happens on its
    own; this one has to be constructed deliberately.
    """
    return {"__type__": type(obj).__name__, "__str__": str(obj)}


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to the one JSON form llmrun hashes.

    Exposed separately from :func:`input_hash` because it is what makes the hash
    inspectable: when a cache miss is unexpected, diffing the canonical JSON of the two
    inputs shows exactly which field moved — and with the tagged fallback that diff now
    names the type as well as the text, so a ``Decimal``/``str`` mix-up reads as one.

    The output of this function is **key material only** — it is hashed, never sent. That
    is worth stating because it was briefly not true: for one revision the same function
    also rendered the payload a model reads, so tagging the fallback to stop
    ``Decimal("1")`` colliding with ``"1"`` silently rewrote every prompt containing an
    exotic type and inflated the cost projection measured from its length.
    :func:`prompt_json` now owns that job, and the two are separate precisely because a key
    and a prompt want opposite things from a fallback.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_tagged,
    )


def prompt_json(value: Any) -> str:
    """Render a payload as the text a model will actually read.

    Same canonical shape as :func:`canonical_json` — sorted keys, no insignificant
    whitespace, real UTF-8 — but with a plain ``str()`` fallback rather than the type tag.

    The two functions exist because one function was doing two jobs whose requirements point
    in opposite directions. A hash key must never merge two distinct values, which is what
    the tag is for. Prompt text has no such requirement and an opposite preference: a model
    reading ``{"amount":{"__str__":"1.50","__type__":"Decimal"}}`` is being shown llmrun's
    bookkeeping instead of the number, and paying for the extra tokens. Splitting them keeps
    the idempotency key collision-free without making every prompt worse.

    Nothing here feeds a cache key, so ``str()`` is safe: the hazards documented for
    :func:`_tagged` — unstable ordering in a ``set``, a memory address in a default
    ``__repr__`` — cost a wrong-looking prompt rather than a wrong cache hit, and both are
    already reasons to reduce a payload to primitives before it gets this far.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def input_hash(step: str, payload: Any, model: str, params: Mapping[str, Any]) -> str:
    """Content address of one unit of work: ``sha256`` hex over canonical JSON.

    All four components are part of the key. ``model`` and ``params`` are included because
    the same prompt sent to a different model, or at a different temperature, is a
    different call that must not be served from the previous one's cache entry; ``step``
    is included so two steps that happen to share a payload keep separate ledger entries.

    ``step()`` hashes non-LLM work by passing ``model="-"`` and ``params={}`` — the same
    function, so resumption and cache lookup can never disagree about what "same input"
    means.
    """
    document = {"step": step, "payload": payload, "model": model, "params": dict(params)}
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
