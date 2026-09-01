"""The provider seam: a two-member protocol and a factory.

llmrun's core carries **zero** provider dependencies. Everything the engine needs from a
model is expressed by :class:`Provider`, and the only import of the ``anthropic`` SDK in
the whole package happens lazily — inside :func:`get_provider`, and inside
:mod:`llmrun.providers.anthropic` itself. That is what lets a reviewer clone the repo,
run ``make demo`` against the echo provider, and never think about an API key.

The protocol is deliberately narrow. Everything that makes llmrun interesting —
checkpointing, budget enforcement, the idempotency ledger, retry policy — lives above
this line, in :mod:`llmrun.context` and :mod:`llmrun.worker`. A provider's entire job is
"turn a prompt into an :class:`~llmrun.types.LLMResult`, or raise a classified error".
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from llmrun.types import LLMResult

__all__ = ["PROVIDER_NAMES", "LLMResult", "Provider", "get_provider"]

logger = logging.getLogger("llmrun.providers.base")

#: Every provider name :func:`get_provider` accepts. The CLI uses this for
#: ``--provider`` choices so the two can never drift apart.
PROVIDER_NAMES: tuple[str, ...] = ("echo", "anthropic")


@runtime_checkable
class Provider(Protocol):
    """A synchronous text-completion backend.

    Implementations must honour three rules, all of which exist because the caller —
    not the provider — owns cost accounting:

    1. **Never retry internally.** A silent retry bills the account twice while the
       caller records one call, so the ledger in ``llm_calls`` stops being the truth.
       Retry policy belongs to :mod:`llmrun.context`, driven by the disposition table.
    2. **Classify failures.** Raise :class:`~llmrun.errors.TransientError` for anything
       worth retrying, :class:`~llmrun.errors.PolicyRefusalError` when the model refused
       (retrying a refusal just burns money), and
       :class:`~llmrun.errors.PermanentError` for everything else.
    3. **Report real token counts.** ``tokens_in``/``tokens_out`` feed
       :func:`llmrun.cost.cost_of` directly; guessing here corrupts every budget.

    ``isinstance(obj, Provider)`` works (the protocol is runtime-checkable) and checks
    for the presence of ``name`` and ``complete``. ``issubclass`` does not — that is a
    CPython restriction on protocols with non-method members, not an oversight.

    **The ``params`` contract.** Two different senders share that one keyword channel, and
    an adapter has to tell them apart:

    * **Reserved keys** are the framework's own bookkeeping, injected by
      :meth:`llmrun.context.RunContext.llm` on every call whether the caller asked for them
      or not. Today that is exactly one key, ``step``, carrying the name of the step being
      served — :class:`~llmrun.providers.echo.EchoProvider` routes its test hooks by it,
      and a real adapter must swallow it, since no API has a ``step`` parameter. Drop
      these *silently*: they arrive on every call, so logging them is pure noise.
    * **Caller keys** are whatever the pipeline author typed into ``ctx.llm(...)``.

    An adapter is expected to forward caller keys through an **allowlist**, not blindly:
    blind ``**params`` passthrough hands the reserved keys to the SDK too, which is a
    ``TypeError`` or an HTTP 400 on every call. But a caller key that falls outside the
    allowlist must not vanish quietly — the author is paying for a call that is not doing
    what they asked, and the parameter is already baked into the idempotency digest, so
    the replayed answer will look like it honoured a setting it never saw. Log a WARNING
    naming the dropped keys. :mod:`llmrun.providers.anthropic` is the worked example;
    its ``_FORWARDED_PARAMS`` and ``_RESERVED_PARAMS`` are the two halves of this rule.
    """

    #: Stable identifier for logs and the ``--provider`` flag, e.g. ``"echo"``.
    name: str

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        payload: str,
        max_tokens: int,
        **params: Any,
    ) -> LLMResult:
        """Run one completion.

        Args:
            model: Model identifier. Must be echoed back on the result, because the
                cost ledger prices the model that actually served the request.
            prompt: The instruction — stable across calls for a given step, and so a
                good prompt-cache prefix.
            payload: The per-call data the instruction operates on.
            max_tokens: Hard ceiling on generated tokens.
            **params: Provider-specific request options, plus the framework's reserved
                keys (``step``). Unknown keys are dropped rather than forwarded blindly,
                so the framework can pass its own bookkeeping through without breaking a
                real SDK call — but a dropped *caller* key must be logged at WARNING. See
                the params contract in the class docstring above.
        """
        ...


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    ``"echo"`` costs nothing and needs nothing. ``"anthropic"`` imports the optional
    SDK *here* rather than at module scope, so a core install with no extras can still
    import :mod:`llmrun.providers` and run the whole test suite.

    Args:
        name: ``"echo"`` or ``"anthropic"``; case- and whitespace-insensitive.
        **kwargs: Forwarded to the provider's constructor (e.g. ``model=``,
            ``latency_seconds=``).

    Raises:
        ValueError: The name is not a known provider.
        ImportError: ``"anthropic"`` was requested but the extra is not installed.
    """
    key = name.strip().lower() if isinstance(name, str) else ""

    if key == "echo":
        from llmrun.providers.echo import EchoProvider

        return EchoProvider(**kwargs)

    if key == "anthropic":
        # Deliberately inside the function: importing this module pulls in the
        # `anthropic` SDK, which is an optional extra. Keeping it here is the whole
        # reason `import llmrun` works on a machine that has never heard of Anthropic.
        from llmrun.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)

    raise ValueError(f"unknown provider {name!r}; expected one of {', '.join(PROVIDER_NAMES)}")
