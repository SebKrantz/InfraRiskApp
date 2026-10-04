"""The assistant's model table: providers, models, reasoning effort, processing tiers.

The ONE place they are written down; /api/assistant/meta serves it to the panel and the
chat endpoint validates a request against it. Keep in step with AGUI
backend/agui/config.py (PROVIDERS, EFFORT_LEVELS, EFFORT_API_DEFAULT, SERVICE_TIERS,
effort_default), so this assistant offers what AGUI's "Models & execution" popover does.
Checked against each vendor's model list on 2026-10-04.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from .. import config

log = logging.getLogger("infrarisk.assistant")

# In display order; the first model is the provider's default unless an env override
# (config.ASSISTANT_MODEL_ENV) names another model of the same list. A model that is not
# in the list (an old ID in an env file or a saved choice) is never used.
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Claude",
        "models": ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-4-5"],
        "default": "claude-sonnet-5-5",
    },
    "gemini": {
        "label": "Gemini",
        "models": ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"],
        "default": "gemini-3.8-flash",
    },
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"],
        "default": "gpt-6-luna",
    },
}

# The reasoning levels a request may pick, by provider ("*") with per-model exceptions; an
# empty list means the model has no effort control. Claude takes `output_config.effort`,
# GPT-6 `reasoning.effort`, Gemini 3 `thinking_level`. Haiku 4.5 does not take the
# parameter; Gemini 3.8 Flash and 3.1 Pro answer `minimal` with a 400, only Flash-Lite takes it.
EFFORT_LEVELS: dict[str, dict[str, list[str]]] = {
    "anthropic": {"*": ["low", "medium", "high", "xhigh", "max"], "claude-haiku-4-5": []},
    "gemini": {
        "*": ["low", "medium", "high"],
        "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
    },
    "openai": {"*": ["low", "medium", "high", "xhigh", "max"]},
}
# What a model runs at when no level is sent (the vendor's API default; Opus 5.5 lowered it
# to medium).
EFFORT_API_DEFAULT = {
    "anthropic": "high",
    "gemini": "high",
    "openai": "medium",
    "claude-opus-5-5": "medium",
}
# Models whose default level is SENT when the request picks none (AGUI's platform default
# for GPT-6 Luna, its smallest tier). For every other model "Default" sends nothing.
EFFORT_SENT = {"gpt-6-luna": "high"}
# Processing tiers by provider. Flex (OpenAI, Gemini) is half price, best-effort and
# slower; Claude has no flex tier.
SERVICE_TIERS: dict[str, list[str]] = {
    "anthropic": ["standard"],
    "gemini": ["standard", "flex"],
    "openai": ["standard", "flex"],
}


def keys() -> dict[str, bool]:
    """Which providers have an API key (availability only, never the key)."""
    return {
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "gemini": bool(config.GEMINI_API_KEY),
        "openai": bool(config.OPENAI_API_KEY),
    }


def default_provider() -> str | None:
    """ASSISTANT_DEFAULT_PROVIDER if keyed, else the first keyed of anthropic, gemini, openai."""
    avail = [pid for pid in PROVIDERS if keys().get(pid)]
    if config.ASSISTANT_DEFAULT_PROVIDER in avail:
        return config.ASSISTANT_DEFAULT_PROVIDER
    return avail[0] if avail else None


_MODEL_VAR = {"anthropic": "ANTHROPIC_MODEL", "gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL"}
_warned: set[tuple[str, str]] = set()


def default_model(provider: str) -> str:
    """The model a provider opens on: its *_MODEL variable if that names one of the
    provider's models in the table, else the table's default. Any other value is ignored,
    with one logged warning per (variable, value)."""
    spec = PROVIDERS[provider]
    override = config.ASSISTANT_MODEL_ENV.get(provider, "")
    if not override:
        return spec["default"]
    if override in spec["models"]:
        return override
    var = _MODEL_VAR[provider]
    if (var, override) not in _warned:
        _warned.add((var, override))
        log.warning(
            "%s=%r is not one of %s's models (%s); using %s",
            var, override, spec["label"], ", ".join(spec["models"]), spec["default"],
        )
    return spec["default"]


def effort_levels(provider: str, model: str) -> list[str]:
    """The levels `model` of `provider` takes ([] for none)."""
    spec = EFFORT_LEVELS.get(provider, {})
    return list(spec.get(model, spec.get("*", [])))


def effort_default(provider: str, model: str) -> str | None:
    """The level shown as "Default (x)"; None when the model has no effort control."""
    if not effort_levels(provider, model):
        return None
    return (
        EFFORT_SENT.get(model)
        or EFFORT_API_DEFAULT.get(model)
        or EFFORT_API_DEFAULT.get(provider)
    )


def service_tiers(provider: str) -> list[str]:
    return list(SERVICE_TIERS.get(provider, ["standard"]))


def meta() -> dict[str, Any]:
    """The table for the panel (availability booleans only, never the keys)."""
    avail = keys()
    providers = [
        {
            "id": pid,
            "label": spec["label"],
            "available": avail.get(pid, False),
            "models": list(spec["models"]),
            "default_model": default_model(pid),
            "efforts": {m: effort_levels(pid, m) for m in spec["models"]},
            "effort_default": {m: effort_default(pid, m) for m in spec["models"]},
            "tiers": service_tiers(pid),
        }
        for pid, spec in PROVIDERS.items()
    ]
    return {
        "available": any(p["available"] for p in providers),
        "default_provider": default_provider(),
        "providers": providers,
    }


class Choice(NamedTuple):
    provider: str
    model: str
    effort: str | None  # the level to SEND; None sends nothing (the model's own default)
    service_tier: str  # "standard" | "flex"


def choose(
    provider: str | None,
    model: str | None,
    effort: str | None = None,
    service_tier: str | None = None,
) -> Choice:
    """Validate a request's choice against the table. A model the provider does not
    offer runs at the provider's default model, a level the model does not take is
    dropped (its default applies), flex on a provider without it runs standard. Raises
    ValueError for a missing or unavailable provider."""
    provider = provider or default_provider()
    if not provider:
        raise ValueError("no assistant provider configured")
    if provider not in PROVIDERS or not keys().get(provider):
        raise ValueError(f"provider {provider!r} is not available")
    if model not in PROVIDERS[provider]["models"]:
        fallback = default_model(provider)
        if model:  # a model that was named but is not on offer; none named is the normal case
            log.warning("chat: model %r is not offered for %s; using %s", model, provider, fallback)
        model = fallback
    level = effort if effort in effort_levels(provider, model) else None
    flex = service_tier == "flex" and "flex" in service_tiers(provider)
    return Choice(provider, model, level or EFFORT_SENT.get(model), "flex" if flex else "standard")
