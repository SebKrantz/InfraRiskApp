"""Provider selection. A provider is offered iff its API key is configured."""

from __future__ import annotations

from ... import config


def available() -> dict[str, bool]:
    return {
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "gemini": bool(config.GEMINI_API_KEY),
    }


def get(name: str):
    """Instantiate the named provider adapter (import deferred: heavy SDKs)."""
    if not available().get(name):
        raise ValueError(f"provider {name!r} is not configured (no API key)")
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"unknown provider {name!r}")
