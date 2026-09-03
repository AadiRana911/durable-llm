"""Provider adapters: one real (Anthropic), one deterministic fake (echo).

Importing this package is free of provider SDKs.
:class:`~durable_llm.providers.anthropic.AnthropicProvider` is safe to import because the
``anthropic`` SDK is only pulled in when the provider is *constructed* — see
:func:`durable_llm.providers.anthropic._import_sdk`.
"""

from __future__ import annotations

from durable_llm.providers.anthropic import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from durable_llm.providers.anthropic import AnthropicProvider
from durable_llm.providers.base import PROVIDER_NAMES, LLMResult, Provider, get_provider
from durable_llm.providers.echo import EchoProvider

__all__ = [
    "ANTHROPIC_DEFAULT_MODEL",
    "PROVIDER_NAMES",
    "AnthropicProvider",
    "EchoProvider",
    "LLMResult",
    "Provider",
    "get_provider",
]
