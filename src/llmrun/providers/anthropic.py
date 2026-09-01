"""The real provider: a thin adapter over the ``anthropic`` SDK.

Three design decisions carry the weight here.

**The SDK is imported lazily.** ``anthropic`` is an optional extra
(``pip install llmrun[anthropic]``); the core, the tests and the demo never touch it.
The import lives inside :func:`_import_sdk`, so importing this module on a machine
without the extra is fine — you only pay when you construct the provider, and you get a
message that names the install command instead of a bare ``ModuleNotFoundError``.

**The adapter never retries.** See :meth:`AnthropicProvider.complete`; this is the whole
reason the module is as small as it is.

**The adapter bounds its own wait.** The request is issued from inside the transaction
that holds the run row's lock, so "how long may one call hang" is a database question
before it is a latency question. See :data:`DEFAULT_TIMEOUT_SECONDS`.

Note on the module name: this file is ``llmrun.providers.anthropic``, but Python 3's
absolute imports mean ``import anthropic`` inside it resolves to the top-level SDK, not
to itself. The two live under different keys in ``sys.modules`` and never collide.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

from llmrun.errors import LLMRunError, PermanentError, PolicyRefusalError, TransientError
from llmrun.types import LLMResult

__all__ = ["DEFAULT_MODEL", "DEFAULT_TIMEOUT_SECONDS", "AnthropicProvider"]

logger = logging.getLogger("llmrun.providers.anthropic")

#: Default model. Overridable per provider instance and per call.
DEFAULT_MODEL = "claude-sonnet-5"

#: Per-request timeout in seconds, used when the caller does not name one.
#:
#: The SDK's own default is the wrong order of magnitude for this engine. As measured on
#: ``anthropic`` 0.121.0 — the floor of the ``anthropic>=0.121,<1`` range this repo's
#: ``[anthropic]`` extra allows, so a newer 0.x may differ and this constant is the thing
#: that stops it mattering — the client defaults to
#: ``httpx.Timeout(connect=5.0, read=600, write=600, pool=600)``: ten minutes of read
#: timeout, against a ``worker.DEFAULT_LEASE_SECONDS`` of 60.
#:
#: That gap is not merely untidy. ``context._billed_call`` issues the request *inside* the
#: transaction that took the run row's ``FOR UPDATE`` lock — deliberately, because that is
#: what stops two workers buying the same completion — so a request the server never
#: answers pins a Postgres row lock and a pooled connection for the SDK's full ten
#: minutes, nine of them after the lease it was working under had already expired.
#:
#: 45 seconds sits strictly under the lease with room to spare: the *first* call of a step
#: starts from a lease ``context._start_step`` has just refreshed, so even a call that runs
#: the clock all the way out leaves roughly fifteen seconds inside the leased window for the
#: ledger insert, the spend update and the commit that turn the timeout into a *recorded*
#: attempt rather than a lost one. That headroom is per step, not per call: the lease is
#: renewed on forward progress, so a step that issues four slow calls back to back still
#: outlives its window, and is meant to — that case is the sweeper's job, and the point of
#: this constant is that no *single* hang can hold the row lock long enough to reach it.
#:
#: It is not a quality-of-service knob — the SDK's
#: ``APITimeoutError`` classifies as :class:`~llmrun.errors.TransientError`, so hitting it
#: hands the call straight to the retry policy in :mod:`llmrun.context` (backoff, jitter,
#: every attempt written down), which is strictly better than stalling behind a lock. A
#: step that legitimately needs longer needs a longer lease too, so raise ``timeout=`` and
#: ``lease_seconds`` together rather than either alone.
#:
#: Not derived from ``worker.DEFAULT_LEASE_SECONDS`` at import time on purpose: the
#: provider seam sits *below* the engine and imports nothing from it, which is what keeps
#: ``get_provider`` usable without dragging the worker into the module graph.
DEFAULT_TIMEOUT_SECONDS = 45.0

#: SDK exception classes that mean "try again later". Resolved by name at runtime so a
#: minor SDK version that adds or renames a class degrades to the status-code check
#: below instead of raising ``AttributeError`` from inside the error handler — the
#: single worst place for a new failure to appear.
_TRANSIENT_EXCEPTION_NAMES = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
)

#: Backstop for the above: HTTP statuses that are retryable regardless of which class
#: the SDK chose to raise. 529 is Anthropic's "overloaded".
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Request options forwarded to ``messages.create``. An allowlist rather than
#: ``**params`` passthrough: the framework routes its own bookkeeping keys (a step name,
#: say) through the same ``**params``, and handing those to the SDK would raise
#: ``TypeError`` or provoke a 400 on every call.
#:
#: ``temperature``, ``top_p`` and ``top_k`` are deliberately **absent**. The current
#: generation — including this adapter's default model — rejects a non-default value for
#: any of them with an HTTP 400, which :meth:`_classify` correctly reads as a
#: ``PermanentError``: a hard, non-retryable run failure for passing a parameter that used
#: to be ordinary. Dropping them turns that failure into a no-op. Route determinism and
#: variance through the prompt, or through ``output_config``.
_FORWARDED_PARAMS = frozenset(
    {
        "metadata",
        "output_config",
        "stop_sequences",
        "thinking",
    }
)

#: Keys the *framework* injects into ``**params`` for its own use, which this adapter is
#: expected to swallow. ``context._billed_call`` sets ``step`` on every single call so the
#: echo provider can route its test hooks by step name; it is not, and never was, an API
#: parameter.
#:
#: They are tracked separately from the caller's keys because the two deserve opposite
#: treatment. A reserved key being dropped is the system working as designed and must stay
#: silent — warning once per call would make the log useless. A *caller's* key being
#: dropped is the caller's request quietly not happening, and :meth:`_request_options`
#: warns about exactly those. Keep this in step with ``context._billed_call``: a bookkeeping
#: key added there and not added here starts producing a warning on every call.
_RESERVED_PARAMS = frozenset({"step"})


def _import_sdk() -> ModuleType:
    """Import the optional ``anthropic`` SDK, or explain how to get it."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the install profile
        raise ImportError(
            "the anthropic provider requires the optional SDK; "
            "install it with: pip install llmrun[anthropic]"
        ) from exc
    return anthropic


class AnthropicProvider:
    """Calls the Anthropic Messages API and maps its failures onto llmrun's taxonomy.

    Args:
        model: Default model when a call does not name one.
        api_key: Passed to the SDK client. Omit to use the SDK's own credential
            resolution (``ANTHROPIC_API_KEY``, then the other sources it supports).
        timeout: Per-request timeout in seconds. ``None`` means
            :data:`DEFAULT_TIMEOUT_SECONDS` — *not* "no timeout", and not the SDK's own
            ten-minute default; pass a number to override it. A caller who genuinely wants
            the SDK's behaviour has to ask for it by value, because the failure mode of
            getting it by accident is a wedged row lock rather than a slow call.
        client: A pre-built SDK client, for tests and for callers who need custom
            transport. Supplying one skips client construction entirely — including the
            timeout above, which is a property of the client, not of the request.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        # Import eagerly so a missing extra surfaces when the worker starts with
        # `--provider anthropic`, not thirty seconds into someone's pipeline.
        self._sdk = _import_sdk()
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key
        # Resolved here rather than at the call site so `provider._timeout` is always the
        # number that will actually be enforced. Substituted only for None: an explicit
        # value is stored verbatim, so a caller who hands us an `httpx.Timeout` (finer
        # grained than the annotation admits, but the SDK accepts it) keeps it intact.
        self._timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
        self._client = client

    # -- public API ---------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The SDK client, built on first use.

        Construction is deferred because it resolves credentials: building it in
        ``__init__`` would make merely *selecting* this provider fail on a box that has
        the SDK but no key, even for a command that never calls the API.
        """
        if self._client is None:
            kwargs: dict[str, Any] = {
                # Deliberate, and the reason this adapter exists in this shape: the SDK
                # retries 408/409/429/5xx twice by default. That retry is invisible to
                # us — we would record one llm_calls row for two or three billed
                # requests, so `spent_usd` would understate reality and a budget cap
                # could be blown right past. Worse, it would break "never make a second
                # call before the first is durably recorded". Retry policy lives in
                # context.py, driven by DISPOSITIONS (exponential backoff + jitter,
                # 5 attempts for TransientError), where each attempt is written down.
                "max_retries": 0,
                # Always set, never conditional: leaving it off would hand the request
                # back to the SDK's ten-minute read timeout, which is the exact thing
                # DEFAULT_TIMEOUT_SECONDS exists to prevent.
                "timeout": self._timeout,
            }
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = self._sdk.Anthropic(**kwargs)
        return self._client

    def complete(
        self,
        *,
        model: str = "",
        prompt: str = "",
        payload: str = "",
        max_tokens: int = 1024,
        **params: Any,
    ) -> LLMResult:
        """Run one completion. Exactly one HTTP request, or an exception.

        The prompt becomes the system prompt and the payload the user turn, which keeps
        the per-step instruction as a stable prefix — the shape prompt caching rewards.
        When there is no payload the prompt moves into the user turn instead, because
        the API requires a non-empty first user message.
        """
        effective_model = model or self.model
        payload_text = payload if isinstance(payload, str) else str(payload)

        request: dict[str, Any] = {
            "model": effective_model,
            "max_tokens": max_tokens,
            **self._request_options(params),
        }
        if payload_text.strip():
            if prompt:
                request["system"] = prompt
            request["messages"] = [{"role": "user", "content": payload_text}]
        else:
            request["messages"] = [{"role": "user", "content": prompt}]

        try:
            response = self.client.messages.create(**request)
        except LLMRunError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc

        return self._to_result(response, requested_model=effective_model)

    # -- internals ----------------------------------------------------------------

    def _request_options(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filter caller params down to real SDK request options.

        ``ctx.llm(**params)`` is documented as passing params through to the provider, and
        every key that does not survive :data:`_FORWARDED_PARAMS` breaks that promise. The
        drop itself is correct — see the allowlist for why ``temperature`` is on the wrong
        side of it — but doing it quietly is not: ``ctx.llm(..., temperature=0)`` used to
        raise nothing, change nothing, and say nothing above DEBUG, so a caller could hash
        the parameter into the ledger key, pay for the call and never learn that the model
        had not been asked for it.

        So the two kinds of dropped key are separated. :data:`_RESERVED_PARAMS` is the
        framework talking to itself and stays silent; anything else came from a human who
        typed it expecting an effect, and gets a WARNING that names it. WARNING rather than
        ERROR because the request is still valid and worth sending — this is a
        misconfiguration to fix, not a reason to fail a run and lose the work before it.
        """
        forwarded = {key: value for key, value in params.items() if key in _FORWARDED_PARAMS}
        dropped = sorted(set(params) - _FORWARDED_PARAMS - _RESERVED_PARAMS)
        if dropped:
            logger.warning(
                "dropping unsupported ctx.llm() params, which had no effect on the "
                "request: %s (this adapter forwards only: %s)",
                ", ".join(dropped),
                ", ".join(sorted(_FORWARDED_PARAMS)),
            )
        return forwarded

    def _to_result(self, response: Any, *, requested_model: str) -> LLMResult:
        """Turn an SDK response into an :class:`LLMResult`, or raise.

        A refusal arrives as a perfectly successful HTTP 200 with
        ``stop_reason == "refusal"`` and often an empty ``content`` list, so the check
        has to happen *before* anything reads the text. Code that indexes
        ``content[0]`` optimistically is the classic way to turn a policy refusal into
        an ``IndexError`` and, one ``except`` later, into a pointless retry.
        """
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise PolicyRefusalError(self._refusal_message(response))

        text = "".join(
            block.text
            for block in getattr(response, "content", None) or ()
            if getattr(block, "type", None) == "text"
        )

        if stop_reason == "max_tokens":
            # Not an error: the caller's schema validation decides whether a truncated
            # answer is usable, and a MalformedOutputError from there routes into the
            # repair loop rather than a blind retry of the same request.
            logger.warning("response truncated at max_tokens (model=%s)", requested_model)

        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            model=getattr(response, "model", None) or requested_model,
            tokens_in=_as_token_count(getattr(usage, "input_tokens", 0)),
            tokens_out=_as_token_count(getattr(usage, "output_tokens", 0)),
        )

    @staticmethod
    def _refusal_message(response: Any) -> str:
        """Build a refusal message from ``stop_details`` when the API supplies one."""
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        parts = ["provider refused the request"]
        if category:
            parts.append(f"category={category}")
        if explanation:
            parts.append(str(explanation))
        return "; ".join(parts)

    def _classify(self, exc: Exception) -> LLMRunError:
        """Map an SDK exception onto the failure taxonomy.

        Everything unrecognised becomes :class:`PermanentError`, which is the honest
        default: an error we cannot name is an error we have no reason to believe a
        retry would fix, and retrying it would spend money to learn nothing.
        """
        transient = tuple(
            cls
            for cls in (getattr(self._sdk, name, None) for name in _TRANSIENT_EXCEPTION_NAMES)
            if isinstance(cls, type) and issubclass(cls, BaseException)
        )
        if transient and isinstance(exc, transient):
            return TransientError(f"{type(exc).__name__}: {exc}")

        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and (status in _TRANSIENT_STATUS_CODES or status >= 500):
            return TransientError(f"HTTP {status}: {exc}")

        return PermanentError(f"{type(exc).__name__}: {exc}")


def _as_token_count(value: Any) -> int:
    """Coerce a usage field to a non-negative int, tolerating ``None``."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
