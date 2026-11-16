"""Provider selection. A provider is offered iff its API key is configured, or the request
brings the user's own key (bring your own key: used for that request only)."""

from __future__ import annotations

from .. import models


def available() -> dict[str, bool]:
    return models.keys()


def get(name: str, api_key: str | None = None):
    """Instantiate the named provider adapter (import deferred: heavy SDKs), with the
    user's own `api_key` when given, else the server's. A fresh adapter per call and
    nothing cached, so a user's key lives only as long as the request that brought it."""
    if not api_key and not available().get(name):
        raise ValueError(f"provider {name!r} is not configured (no API key)")
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(api_key)
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key)
    raise ValueError(f"unknown provider {name!r}")
