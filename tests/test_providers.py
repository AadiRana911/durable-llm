"""The Anthropic adapter's silent-failure paths, tested without the SDK installed.

Every bug this file guards against is invisible in a normal run and expensive in a real one:
a refusal that falls through as an empty answer instead of routing to a human, a request
parameter the API stopped accepting that turns every call into a hard non-retryable failure,
a caller's parameter dropped without a word, and an SDK-level retry that bills twice for one
ledger row. None of them shows up unless something actually exercises the adapter.

``anthropic`` is an optional extra and is not installed for the test suite, so
:func:`_import_sdk` is patched with a stand-in module. That is not a compromise: the adapter
resolves SDK exception classes by name at runtime precisely so a missing or differently
shaped SDK degrades instead of raising from inside the error handler, and a fake module is
the only way to test that decision.

It does mean the stand-in has to come in two shapes, and both are needed:

* :func:`_stub_sdk` (autouse) is **empty** — no exception classes at all. That is the
  degraded SDK, where ``_classify`` has nothing to match on and the HTTP status code is the
  whole signal.
* :func:`populated_sdk` carries classes actually *named* ``RateLimitError``,
  ``APIConnectionError``, ``APITimeoutError`` and ``InternalServerError``, and deliberately
  gives none of them a ``status_code``. That is the healthy SDK, and it is the shape the
  name-resolution path in ``_classify`` was written for. With only the empty stub in the
  suite, every classification test was answered by the status-code backstop and the
  ``getattr(self._sdk, name)`` lookup never resolved a single class — so a typo in
  :data:`_TRANSIENT_EXCEPTION_NAMES`, or a lookup that silently returned nothing, degraded
  every rate limit into a ``PermanentError`` and nothing here noticed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from durable_llm.errors import PermanentError, PolicyRefusalError, TransientError
from durable_llm.providers import anthropic as adapter


class _FakeMessages:
    """Records the request it was handed and replays a canned response."""

    def __init__(self, response: Any, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        self.messages = _FakeMessages(response, error)


def _response(**fields: Any) -> SimpleNamespace:
    """A response object shaped like the SDK's, with usage attached."""
    defaults: dict[str, Any] = {
        "stop_reason": "end_turn",
        "content": [SimpleNamespace(type="text", text="{}")],
        "model": "claude-sonnet-5",
        "usage": SimpleNamespace(input_tokens=11, output_tokens=7),
        "stop_details": None,
    }
    return SimpleNamespace(**{**defaults, **fields})


@pytest.fixture(autouse=True)
def _stub_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the optional ``anthropic`` package.

    Deliberately carries none of the SDK's exception classes, which is the degraded shape
    :func:`AnthropicProvider._classify` is written to survive: it falls back to the HTTP
    status code rather than raising ``AttributeError`` from inside the error handler.
    """
    monkeypatch.setattr(adapter, "_import_sdk", lambda: SimpleNamespace())


#: The SDK exception classes that mean "try again later", spelled out by hand rather than
#: read back from ``adapter._TRANSIENT_EXCEPTION_NAMES``. Deriving them from the adapter's
#: own tuple would make a typo in that tuple invisible — the fake SDK would grow the same
#: typo and the lookup would keep succeeding — which is precisely the failure this file is
#: here to catch. These are the names the ``anthropic`` SDK publishes; if the SDK renames
#: one, this list is where the rename gets noticed and argued about.
_TRANSIENT_SDK_ERRORS = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
)

#: The classes themselves: plain ``Exception`` subclasses with **no** ``status_code``, on
#: purpose. The status-code backstop must not be able to answer for them, so a passing
#: classification proves the *name* lookup fired and not the fallback underneath it.
_TRANSIENT_CLASSES: dict[str, type[Exception]] = {
    name: type(name, (Exception,), {}) for name in _TRANSIENT_SDK_ERRORS
}


@pytest.fixture
def populated_sdk(monkeypatch: pytest.MonkeyPatch, _stub_sdk: None) -> SimpleNamespace:
    """A stand-in SDK that carries the exception classes, unlike the autouse stub.

    Depends on :func:`_stub_sdk` explicitly rather than relying on autouse ordering, so the
    empty namespace is installed first and this one overwrites it. Getting that backwards
    would leave the tests below running against the empty stub — where they would still pass
    for the wrong reason, via the status-code path, which is exactly the blind spot this
    fixture exists to remove.
    """
    sdk = SimpleNamespace(**_TRANSIENT_CLASSES)
    monkeypatch.setattr(adapter, "_import_sdk", lambda: sdk)
    return sdk


@pytest.fixture
def constructed_clients(monkeypatch: pytest.MonkeyPatch, _stub_sdk: None) -> list[dict[str, Any]]:
    """Records the kwargs every ``sdk.Anthropic(...)`` construction was handed.

    Every other test in this file passes ``client=`` and therefore never builds one, which
    is what left the constructor untested: it is the only place ``max_retries`` and the
    timeout are set, and both are load-bearing rather than cosmetic.
    """
    recorded: list[dict[str, Any]] = []

    class _RecordingAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            recorded.append(kwargs)
            self.messages = _FakeMessages(_response())

    monkeypatch.setattr(
        adapter, "_import_sdk", lambda: SimpleNamespace(Anthropic=_RecordingAnthropic)
    )
    return recorded


def test_a_refusal_becomes_a_policy_error_rather_than_an_empty_answer() -> None:
    """A refusal is an HTTP 200 with no content — the one failure that looks like success.

    Code that reads ``content[0]`` optimistically turns it into an ``IndexError`` and, one
    ``except`` later, into a retry: every attempt billed, every attempt refused. The check
    has to happen before anything reads the text, and the result has to be a *policy* error
    so the disposition table routes the run to ``pending_review`` instead of retrying it.
    """
    client = _FakeClient(
        _response(
            stop_reason="refusal",
            content=[],
            stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        )
    )
    provider = adapter.AnthropicProvider(client=client)

    with pytest.raises(PolicyRefusalError) as caught:
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")

    message = str(caught.value)
    assert "category=cyber" in message
    assert "declined" in message


def test_sampling_parameters_are_dropped_instead_of_being_sent() -> None:
    """``temperature``/``top_p``/``top_k`` are rejected with a 400 by current models.

    ``_classify`` reads a 400 as ``PermanentError``, so forwarding one of these fails the
    whole run, non-retryably, with a diagnosis that points at nothing. Dropping them means a
    caller who passes ``temperature=0.2`` — which the README used to suggest — still gets an
    answer. The cache key still includes the parameter, so this is not a correctness change:
    two calls that differ only by a dropped parameter remain two different cache entries.
    """
    client = _FakeClient(_response())
    provider = adapter.AnthropicProvider(client=client)

    provider.complete(
        prompt="sys",
        payload="hello",
        max_tokens=64,
        step="x",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
    )

    sent = client.messages.requests[0]
    assert set(sent) == {"model", "max_tokens", "system", "messages"}, (
        f"the adapter forwarded a parameter the API rejects: {sorted(sent)}"
    )


def test_allowlisted_request_options_are_still_forwarded() -> None:
    """The allowlist has to keep letting real request options through."""
    client = _FakeClient(_response())
    provider = adapter.AnthropicProvider(client=client)

    provider.complete(
        prompt="sys",
        payload="hello",
        max_tokens=64,
        step="x",
        output_config={"effort": "high"},
        stop_sequences=["\n\n"],
    )

    sent = client.messages.requests[0]
    assert sent["output_config"] == {"effort": "high"}
    assert sent["stop_sequences"] == ["\n\n"]


def test_a_dropped_caller_param_warns_while_a_reserved_one_stays_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The drop is correct; doing it quietly is not, and only one kind may be quiet.

    ``ctx.llm(..., temperature=0)`` is hashed into the ledger key, billed for, and then
    never sent — so without a warning the caller pays for a call the model was not asked to
    make differently, and has nothing anywhere telling them so. DEBUG is not enough: the CLI
    ships at INFO, so a DEBUG line is a line nobody reads.

    ``step`` is the opposite case and must stay silent. ``context._billed_call`` sets it on
    *every single call* so the echo provider can route its hooks by step name; warning about
    it would put one line of noise per call into the log and train operators to ignore the
    warning that matters. This test fails in both directions — a silenced caller key and a
    newly chatty reserved key — because either one breaks the log's signal-to-noise.
    """
    provider = adapter.AnthropicProvider(client=_FakeClient(_response()))

    with caplog.at_level(logging.WARNING, logger="durable_llm.providers.anthropic"):
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x", temperature=0.2)

    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {[r.message for r in warnings]}"
    assert "temperature" in warnings[0].getMessage(), (
        "the warning must name the parameter that was dropped; 'some params were ignored' "
        "sends the reader back to the source to find out which"
    )

    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="durable_llm.providers.anthropic"):
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")

    assert caplog.records == [], (
        "the framework-reserved 'step' key warned about itself; it is injected on every "
        f"call, so this is one log line per call forever: {[r.message for r in caplog.records]}"
    )


def test_the_constructed_client_never_retries_and_bounds_its_own_wait(
    constructed_clients: list[dict[str, Any]],
) -> None:
    """Two constructor kwargs, both of which fail invisibly if they are ever dropped.

    ``max_retries=0`` is the load-bearing one. The SDK retries 408/409/429/5xx twice by
    default, inside one ``messages.create`` call — so durable_llm would write **one**
    ``llm_calls`` row for up to three billed requests. ``spent_usd`` would understate the
    invoice, a budget cap could be sailed straight past, and the library's central promise
    ("never make a second call before the first is durably recorded") would be false with no
    symptom: the run succeeds, the ledger looks tidy, the bill does not match it. Retry
    policy belongs to ``DISPOSITIONS``, where every attempt gets written down.

    The timeout is the other half. ``context._billed_call`` issues the request *inside* the
    transaction holding the run row's ``FOR UPDATE`` lock, so the SDK's ten-minute read
    timeout would pin that lock — and a pooled connection — for nine minutes after the lease
    expired. Asserting the adapter's own constant rather than a literal keeps this test
    honest if the number is retuned; asserting that it is *passed at all* is the point.
    """
    provider = adapter.AnthropicProvider()

    assert constructed_clients == [], "the client must not be built until it is used"

    provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")

    assert len(constructed_clients) == 1
    kwargs = constructed_clients[0]
    assert kwargs["max_retries"] == 0, (
        "the SDK will retry a 429 twice on its own; those calls are billed and none of them "
        "reach llm_calls, so spend is understated and the budget cap cannot see them"
    )
    assert kwargs["timeout"] == adapter.DEFAULT_TIMEOUT_SECONDS
    assert adapter.DEFAULT_TIMEOUT_SECONDS < 60, (
        "the default timeout must stay under worker.DEFAULT_LEASE_SECONDS, or one hung "
        "request holds the run's row lock past the lease it was working under"
    )
    # No api_key: omitted so the SDK's own credential resolution still applies.
    assert "api_key" not in kwargs

    # Built once and reused — a client rebuilt per call would re-resolve credentials on
    # every request and quietly reintroduce whatever defaults it was constructed without.
    provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")
    assert len(constructed_clients) == 1


def test_an_explicit_timeout_overrides_the_default() -> None:
    """``timeout=None`` means the adapter's default, not "no timeout" — so a real number
    has to survive to the constructor unchanged, or the override is decorative."""
    provider = adapter.AnthropicProvider(timeout=5.0)

    assert provider._timeout == 5.0


@pytest.mark.parametrize("error_name", _TRANSIENT_SDK_ERRORS)
def test_named_sdk_exceptions_classify_as_transient(
    error_name: str, populated_sdk: SimpleNamespace
) -> None:
    """The name-resolution path, which every other test in this file used to skip.

    ``_classify`` looks each of these up with ``getattr(self._sdk, name, None)`` and matches
    on the class. That indirection exists so a minor SDK bump which renames or removes a
    class degrades instead of raising ``AttributeError`` from inside the error handler — but
    the same indirection means a *missed* name fails silently, in the safest-looking way
    possible: the lookup returns ``None``, the class is filtered out, and the exception falls
    through to ``PermanentError``. A rate limit then ends the run instead of being retried.

    None of these fakes carries a ``status_code``, so the backstop below cannot rescue them.
    If this goes red, ``_TRANSIENT_EXCEPTION_NAMES`` and the SDK have drifted apart, and
    every rate limit in production is currently a hard failure.
    """
    failure = _TRANSIENT_CLASSES[error_name]("upstream said no")
    assert not hasattr(failure, "status_code"), "the fake must not be answerable by status code"

    provider = adapter.AnthropicProvider(client=_FakeClient(error=failure))

    with pytest.raises(TransientError) as caught:
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")

    # The class name is carried into the message: "TransientError: retrying" tells an
    # operator nothing about which upstream condition produced it.
    assert error_name in str(caught.value)


def test_the_adapters_transient_names_match_the_sdks() -> None:
    """The list above is written out by hand; this is what ties it to the adapter.

    Without it, an entry deleted from ``_TRANSIENT_EXCEPTION_NAMES`` would simply stop being
    parametrised over and the suite would go quietly greener. Compared as sets rather than
    tuples because order carries no meaning — ``_classify`` matches against all of them.
    """
    assert set(adapter._TRANSIENT_EXCEPTION_NAMES) == set(_TRANSIENT_SDK_ERRORS)


def test_an_unrecognised_sdk_exception_is_permanent_even_with_the_classes_present(
    populated_sdk: SimpleNamespace,
) -> None:
    """The default has to stay ``PermanentError`` when the name lookup *succeeds* too.

    A populated SDK makes the ``isinstance`` check meaningful, and the risk on that path is
    the mirror image of the one above: a match that is too broad (matching ``Exception``, or
    a base class the SDK also uses for 400s) would make every bug retryable — five billed
    repetitions of the same crash.
    """
    provider = adapter.AnthropicProvider(client=_FakeClient(error=ValueError("a real bug")))

    with pytest.raises(PermanentError):
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, TransientError), (503, TransientError), (529, TransientError), (400, PermanentError)],
)
def test_http_status_codes_map_onto_the_taxonomy(status: int, expected: type) -> None:
    """With no SDK exception classes to match on, the status code is the whole signal."""

    class _ApiError(Exception):
        status_code = status

    provider = adapter.AnthropicProvider(client=_FakeClient(error=_ApiError("boom")))

    with pytest.raises(expected):
        provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")


def test_usage_is_read_back_onto_the_result() -> None:
    """Token counts come from the provider, never from the local estimate."""
    provider = adapter.AnthropicProvider(client=_FakeClient(_response()))
    result = provider.complete(prompt="sys", payload="hello", max_tokens=64, step="x")

    assert (result.tokens_in, result.tokens_out) == (11, 7)
    assert result.model == "claude-sonnet-5"
    assert result.text == "{}"
