"""Provider adapters: one real (Anthropic), one deterministic fake (echo).

Importing this package is free of provider SDKs.
:class:`~llmrun.providers.anthropic.AnthropicProvider` is safe to import because the
``anthropic`` SDK is only pulled in when the provider is *constructed* — see
:func:`llmrun.providers.anthropic._import_sdk`.
"""

from __future__ import annotations

from llmrun.providers.anthropic import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from llmrun.providers.anthropic import AnthropicProvider
from llmrun.providers.base import PROVIDER_NAMES, LLMResult, Provider, get_provider
from llmrun.providers.echo import EchoProvider

__all__ = [
    "ANTHROPIC_DEFAULT_MODEL",
    "PROVIDER_NAMES",
    "AnthropicProvider",
    "EchoProvider",
    "LLMResult",
    "Provider",
    "get_provider",
]
