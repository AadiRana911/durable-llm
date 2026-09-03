"""Hash determinism — the property the no-double-billing guarantee is built on.

``input_hash`` is the key of the idempotency store. If it varies between the process that
made a call and the process that resumes after that process was killed, the resumed run
misses the ledger, calls the provider again, and pays twice. The failure is silent: a
cache miss is indistinguishable from ordinary first-time work.

So these tests pin the canonicalisation from both sides:

* **must not change the hash** — mapping key order, JSON formatting, parameter order,
  and (most importantly) the process the hash is computed in;
* **must change the hash** — the step name, the payload, the model, and the params.

The cross-process test is the one that earns its keep. It computes the same hash in two
subprocesses started with different ``PYTHONHASHSEED`` values, which is the classic way an
"obviously deterministic" hash turns out not to be: anything that falls back to ``str()``
of a set, or to Python's salted ``hash()``, produces a different digest in a different
process and nothing in-process ever notices.

There is a third group at the bottom, covering the two renderers the module now exposes.
``canonical_json`` is key material and tags every value JSON cannot represent with its type
name, because a fallback that merges ``Decimal("1")`` with ``"1"`` makes the ledger hand one
call another call's stored answer. ``prompt_json`` is the text a model reads and keeps the
old untagged ``str()`` rendering, because a model shown ``{"__type__":"Decimal"}`` is being
charged tokens for durable_llm's bookkeeping. The tests below assert both halves of that split
directly, so the two functions cannot be quietly merged back into one by someone tidying up
what looks like duplication.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from durable_llm.cache import HASH_ALGORITHM, canonical_json, input_hash, prompt_json

# A payload with some nesting, some non-ASCII, and mixed types — a realistic step payload
# rather than a happy-path string.
PAYLOAD = {
    "thread_id": "thread-42",
    "messages": [
        {"author": "ana", "body": "café ✅", "index": 0},
        {"author": "bo", "body": "rollback shipped", "index": 1},
    ],
    "meta": {"locale": "en-GB", "priority": 2},
}

PARAMS = {"temperature": 0, "top_p": 1}

# Computes one hash and prints it. Run in a subprocess with a hostile PYTHONHASHSEED to
# prove the digest does not depend on anything process-local.
_CROSS_PROCESS_SCRIPT = """
import json, sys
from durable_llm.cache import input_hash

step, payload_json, model, params_json = sys.argv[1:5]
sys.stdout.write(input_hash(step, json.loads(payload_json), model, json.loads(params_json)))
"""


def _hash_in_subprocess(hash_seed: str) -> str:
    """Compute the reference hash in a fresh interpreter with ``PYTHONHASHSEED`` set."""
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CROSS_PROCESS_SCRIPT,
            "extract_decisions",
            json.dumps(PAYLOAD),
            "claude-sonnet-5",
            json.dumps(PARAMS),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    return result.stdout.strip()


# --------------------------------------------------------------------------------------
# shape and determinism
# --------------------------------------------------------------------------------------


def test_hash_is_a_sha256_hex_digest() -> None:
    digest = input_hash("extract", PAYLOAD, "claude-sonnet-5", PARAMS)

    assert HASH_ALGORITHM == "sha256"
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_repeated_calls_agree() -> None:
    """The cheapest property, and the one everything else assumes."""
    first = input_hash("extract", PAYLOAD, "claude-sonnet-5", PARAMS)
    second = input_hash("extract", PAYLOAD, "claude-sonnet-5", PARAMS)

    assert first == second


def test_hash_is_stable_across_processes_and_hash_seeds() -> None:
    """The load-bearing one: a resumed worker is a *different process*.

    Two subprocesses with deliberately different ``PYTHONHASHSEED`` values must agree with
    each other and with this process. Anything that leaked Python's salted ``hash()`` — or
    the iteration order of a set — into the digest fails here and nowhere else.
    """
    in_process = input_hash("extract_decisions", PAYLOAD, "claude-sonnet-5", PARAMS)

    assert _hash_in_subprocess("0") == in_process
    assert _hash_in_subprocess("12345") == in_process


# --------------------------------------------------------------------------------------
# things that must NOT change the hash
# --------------------------------------------------------------------------------------


def test_dict_key_order_does_not_change_the_hash() -> None:
    """Same document, different insertion order — the same logical input."""
    ordered = {"alpha": 1, "beta": 2, "gamma": {"x": 10, "y": 20}}
    shuffled = {"gamma": {"y": 20, "x": 10}, "beta": 2, "alpha": 1}

    assert list(ordered) != list(shuffled)  # the dicts really are ordered differently
    assert input_hash("s", ordered, "m", {}) == input_hash("s", shuffled, "m", {})


def test_param_key_order_does_not_change_the_hash() -> None:
    assert input_hash("s", PAYLOAD, "m", {"temperature": 0, "top_p": 1}) == input_hash(
        "s", PAYLOAD, "m", {"top_p": 1, "temperature": 0}
    )


def test_irrelevant_whitespace_does_not_change_the_hash() -> None:
    """Re-serialising a payload with different formatting cannot change its key.

    A payload frequently arrives as a JSON document from somewhere else, and whether that
    document was pretty-printed is not a property of the request. Both texts below decode
    to the same object, so both must hash the same — which is what ``separators=(",", ":")``
    in the canonicaliser buys.
    """
    document = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    compact = json.dumps(document, separators=(",", ":"))
    pretty = json.dumps(document, indent=4, sort_keys=True)

    assert compact != pretty
    assert input_hash("s", json.loads(compact), "m", {}) == input_hash(
        "s", json.loads(pretty), "m", {}
    )


def test_mapping_type_does_not_change_the_hash() -> None:
    """A plain dict and an equal dict built another way are the same input."""
    built = dict(sorted(PAYLOAD.items(), reverse=True))

    assert input_hash("s", PAYLOAD, "m", {}) == input_hash("s", built, "m", {})


# --------------------------------------------------------------------------------------
# things that MUST change the hash
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "other"),
    [
        ("step", ("other_step", PAYLOAD, "claude-sonnet-5", PARAMS)),
        ("payload", ("extract", {**PAYLOAD, "thread_id": "thread-43"}, "claude-sonnet-5", PARAMS)),
        ("model", ("extract", PAYLOAD, "claude-opus-5", PARAMS)),
        ("params", ("extract", PAYLOAD, "claude-sonnet-5", {**PARAMS, "temperature": 1})),
        ("extra param", ("extract", PAYLOAD, "claude-sonnet-5", {**PARAMS, "seed": 7})),
    ],
)
def test_a_different_request_hashes_differently(label: str, other: tuple) -> None:
    """Every component of the key is really part of the key.

    ``model`` and ``params`` matter most: the same prompt at a different temperature, or
    against a different model, is a different call. Serving it from the previous call's
    ledger entry would silently return the wrong answer *and* under-report spend.
    """
    baseline = input_hash("extract", PAYLOAD, "claude-sonnet-5", PARAMS)

    assert input_hash(*other) != baseline, f"{label} must be part of the hash"


def test_payload_types_are_distinguished() -> None:
    """Two values that render as the same characters must not share an idempotency key.

    The first two assertions are the cheap ones: ``json.dumps`` already writes ``1`` and
    ``"1"`` differently, so they held under every version of the canonicaliser and exercise
    nothing that was ever broken. They stay because they are the boundary the other pairs
    generalise, not because they are load-bearing.

    The last two are the defect. A value JSON cannot represent goes through the ``default=``
    fallback, and that fallback used to be bare ``str``: ``Decimal("1")`` became the four
    bytes ``"1"``, which is exactly what the string ``"1"`` becomes, and a ``UUID`` became
    its own text. Two genuinely different requests then produced one hash, and the second
    one was served the first one's stored response out of ``llm_calls`` — a *wrong answer*,
    returned confidently, with a ledger row asserting it was a legitimate replay. Nothing
    anywhere else in the suite fails when that fallback is restored; these two lines do.
    """
    assert input_hash("s", 1, "m", {}) != input_hash("s", "1", "m", {})
    assert input_hash("s", [1, 2], "m", {}) != input_hash("s", ["1", "2"], "m", {})

    assert input_hash("s", Decimal("1"), "m", {}) != input_hash("s", "1", "m", {}), (
        "Decimal('1') and the string '1' share an idempotency key: the fallback is "
        "type-blind again, and one call can be served the other's stored response"
    )

    identifier = UUID("00000000-0000-0000-0000-0000000000ff")
    assert input_hash("s", identifier, "m", {}) != input_hash("s", str(identifier), "m", {}), (
        "a UUID and its own text share an idempotency key"
    )


def test_step_hashing_convention_is_distinct_from_llm_hashing() -> None:
    """``ctx.step`` hashes with ``model="-"``; that must not collide with a real model."""
    assert input_hash("fetch", PAYLOAD, "-", {}) != input_hash("fetch", PAYLOAD, "echo", {})


# --------------------------------------------------------------------------------------
# canonical_json
# --------------------------------------------------------------------------------------


def test_canonical_json_is_sorted_compact_and_unicode_preserving() -> None:
    text = canonical_json({"b": 2, "a": "café"})

    assert text == '{"a":"café","b":2}'
    assert " " not in text  # no insignificant whitespace survives


def test_non_json_types_are_type_tagged_rather_than_raising() -> None:
    """The fallback has to do two things at once, and the tag is the half that is new.

    ``Decimal``, ``UUID`` and ``datetime`` turn up in real payloads, and hashing must never
    be the reason a pipeline crashes — so the canonicaliser needs *some* ``default=``. It
    used to be bare ``str``, which satisfied that requirement and quietly broke the more
    important one: ``str`` is type-blind, so distinct values collapsed onto one key and the
    ledger replayed one call's answer into another call's step.

    ``{"__type__": ..., "__str__": ...}`` keeps the crash-proofing and separates the types.
    The literal-text assertion is the point of this test: without it, every remaining
    assertion here still passes under ``default=str`` — the digest is still 64 characters,
    still reproducible, and ``Decimal("1.50")`` still differs from the float ``1.5`` because
    ``"1.50"`` is not ``1.5``. Only the rendering says whether the tag actually shipped.
    """
    payload = {
        "amount": Decimal("1.50"),
        "id": UUID("00000000-0000-0000-0000-0000000000ff"),
        "at": datetime(2026, 3, 1, 9, 0, tzinfo=UTC),
    }

    # The tag, spelled out. Keys inside it sort like any other, so `__str__` precedes
    # `__type__` — pinning the whole rendering also pins sort_keys and the separators.
    rendered = canonical_json({"amount": Decimal("1.50")})
    assert '"__type__":"Decimal"' in rendered, (
        f"the type tag is missing from {rendered!r}; the fallback has reverted to something "
        "type-blind and two different values can now share one idempotency key"
    )
    assert rendered == '{"amount":{"__str__":"1.50","__type__":"Decimal"}}'

    digest = input_hash("s", payload, "m", {})

    assert len(digest) == 64
    assert digest == input_hash("s", payload, "m", {})
    # Carried as text, not coerced to a number: Decimal("1.50") and 1.5 are different inputs.
    assert digest != input_hash("s", {**payload, "amount": 1.5}, "m", {})


def test_prompt_json_renders_the_value_rather_than_its_type_tag() -> None:
    """The other half of the split, and the reason there are two functions instead of one.

    ``prompt_json`` produces *prompt text*: ``context._complete`` hands its output straight
    to the provider as the user turn, and ``budget.project_cost`` sizes the pre-call estimate
    from its length. Neither wants the tag. A model shown
    ``{"amount":{"__str__":"1.50","__type__":"Decimal"}}`` is being asked to read durable_llm's
    bookkeeping instead of the number, and the caller is paying for the extra tokens.

    The two functions differ by one keyword argument and are otherwise identical, which makes
    them look exactly like the duplication a tidy-up removes. Merging them in either
    direction fails here: fold ``prompt_json`` into ``canonical_json`` and the tag appears in
    the prompt; fold ``canonical_json`` into ``prompt_json`` and
    ``test_payload_types_are_distinguished`` goes red as the collision comes back.
    """
    assert prompt_json({"amount": Decimal("1.50")}) == '{"amount":"1.50"}'
    assert "__type__" not in prompt_json({"amount": Decimal("1.50")})

    # Same canonical shape as the hash renderer in every respect except the fallback: sorted
    # keys, no insignificant whitespace, real UTF-8 rather than \\u escapes.
    assert prompt_json({"b": 2, "a": "café"}) == '{"a":"café","b":2}'
    assert prompt_json({"b": 2, "a": "café"}) == canonical_json({"b": 2, "a": "café"})
