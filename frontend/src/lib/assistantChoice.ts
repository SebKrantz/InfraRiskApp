// The assistant's model / effort / execution setting: validation against the table that
// GET /api/assistant/meta serves (backend/app/assistant/models.py, in step with AGUI
// backend/agui/config.py), persistence in localStorage, and the handoff from AGUI.
//
// AGUI opens this app with `?assistant=1&ai_provider=..&ai_model=..` and, only when chosen,
// `&ai_effort=..` and `&ai_tier=flex`. They are read once at module load and taken off the
// address bar (the rest of the query stays); Assistant.tsx applies them when /meta arrives.

import type { AssistantChoice, AssistantMeta, AssistantProvider } from '../types/assistant'

const STORAGE_KEY = 'infrarisk.assistant.choice'

export function effortsFor(provider: AssistantProvider | undefined, model: string): string[] {
  return provider?.efforts?.[model] ?? []
}

export function canFlex(provider: AssistantProvider | undefined): boolean {
  return Boolean(provider?.tiers?.includes('flex'))
}

const usable = (meta: AssistantMeta, id: string | null | undefined) =>
  meta.providers.find((p) => p.id === id && p.available)

/** `raw` checked against the table: anything no longer on offer falls back to the default. */
export function resolveChoice(
  meta: AssistantMeta,
  raw: Partial<AssistantChoice> | null,
): AssistantChoice {
  const provider =
    usable(meta, raw?.provider) ??
    usable(meta, meta.default_provider) ??
    meta.providers.find((p) => p.available) ??
    meta.providers[0]
  const same = provider.id === raw?.provider
  const kept = same && provider.models.includes(raw?.model ?? '')
  const model = kept ? (raw?.model as string) : provider.default_model
  // an effort was picked for the saved model, a tier for the saved provider: they go with it
  const effort = kept && effortsFor(provider, model).includes(raw?.effort ?? '') ? (raw?.effort as string) : null
  return {
    provider: provider.id,
    model,
    effort,
    tier: same && raw?.tier === 'flex' && canFlex(provider) ? 'flex' : 'standard',
  }
}

export function loadChoice(): Partial<AssistantChoice> | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? (JSON.parse(raw) as unknown) : null
    return parsed && typeof parsed === 'object' ? (parsed as Partial<AssistantChoice>) : null
  } catch {
    return null // storage blocked or unreadable: the default applies
  }
}

export function saveChoice(choice: AssistantChoice): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(choice))
  } catch {
    /* storage blocked: the choice lasts until the page closes */
  }
}

/** The user picks another provider: its default model, no effort, Standard if it has no flex. */
export function withProvider(
  meta: AssistantMeta,
  current: AssistantChoice,
  providerId: string,
): AssistantChoice {
  const next = meta.providers.find((p) => p.id === providerId)
  if (!next) return current
  return {
    provider: next.id,
    model: next.default_model,
    effort: null,
    tier: canFlex(next) ? current.tier : 'standard',
  }
}

/** The user picks another model of the provider: an effort it does not take is dropped. */
export function withModel(
  meta: AssistantMeta,
  current: AssistantChoice,
  model: string,
): AssistantChoice {
  const provider = meta.providers.find((p) => p.id === current.provider)
  const keep = current.effort !== null && effortsFor(provider, model).includes(current.effort)
  return { ...current, model, effort: keep ? current.effort : null }
}

// ---- handoff from AGUI ------------------------------------------------------------------

export interface Handoff {
  provider: string | null
  model: string | null
  effort: string | null
  tier: string | null
}

const HANDOFF_PARAMS = ['ai_provider', 'ai_model', 'ai_effort', 'ai_tier'] as const

/** The ai_* parameters of a query string; null when there are none. */
export function parseHandoff(search: string): Handoff | null {
  const q = new URLSearchParams(search)
  const get = (name: string) => q.get(name)?.trim() || null
  const h = {
    provider: get('ai_provider'),
    model: get('ai_model'),
    effort: get('ai_effort'),
    tier: get('ai_tier'),
  }
  return h.provider || h.model || h.effort || h.tier ? h : null
}

/**
 * The setting a handoff makes. With an ai_provider or ai_model it is a COMPLETE setting: AGUI
 * sends ai_effort only when a level was chosen and ai_tier only for flex, so an absent or
 * invalid effort means Default and an absent or unoffered tier means Standard; nothing the app
 * saved earlier survives. The provider must be keyed; a model alone names its provider. With
 * neither (only ai_effort / ai_tier), each valid value changes `current` and the rest stays.
 * Null: nothing in the handoff is usable.
 */
export function applyHandoff(
  meta: AssistantMeta,
  h: Handoff,
  current: AssistantChoice,
): AssistantChoice | null {
  if (!h.provider && !h.model) {
    const provider = meta.providers.find((p) => p.id === current.provider)
    const effort = h.effort && effortsFor(provider, current.model).includes(h.effort) ? h.effort : current.effort
    const tier =
      h.tier === 'standard' || (h.tier === 'flex' && canFlex(provider))
        ? (h.tier as AssistantChoice['tier'])
        : current.tier
    return effort === current.effort && tier === current.tier ? null : { ...current, effort, tier }
  }
  const provider =
    usable(meta, h.provider) ??
    (h.model ? meta.providers.find((p) => p.available && p.models.includes(h.model as string)) : undefined)
  if (!provider) return null
  const model = h.model && provider.models.includes(h.model) ? h.model : provider.default_model
  return {
    provider: provider.id,
    model,
    effort: h.effort && effortsFor(provider, model).includes(h.effort) ? h.effort : null,
    tier: h.tier === 'flex' && canFlex(provider) ? 'flex' : 'standard',
  }
}

/** Read the ai_* parameters off the address bar and remove them, keeping the rest. */
function takeHandoff(): Handoff | null {
  const h = parseHandoff(window.location.search)
  if (!h) return null
  try {
    const url = new URL(window.location.href)
    HANDOFF_PARAMS.forEach((name) => url.searchParams.delete(name))
    window.history.replaceState(window.history.state, '', url)
  } catch {
    /* a locked-down history: the parameters stay in the bar */
  }
  return h
}

// Once, at module load: before the app can rewrite location.search to mirror its own state.
export const HANDOFF: Handoff | null = typeof window === 'undefined' ? null : takeHandoff()
