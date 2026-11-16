// Bring your own key: the user's own Anthropic, Google or OpenAI API key, on servers that
// accept them (GET /api/assistant/meta → `byok`). The key stays in this browser — in
// sessionStorage (gone when the tab closes) unless the user asks to remember it on this
// device (localStorage), and only in memory where storage is blocked. It travels with each
// chat request in the JSON body (not a header, which proxies may log); the server uses it
// for that request and never stores, logs or returns it.

import type { AssistantMeta } from '../types/assistant'

const PREFIX = 'infrarisk.assistant.key.'
const memory = new Map<string, string>()

export const KEY_HINTS: Record<string, string> = {
  anthropic: 'sk-ant-…',
  gemini: 'AIza…',
  openai: 'sk-…',
}

function storage(kind: 'session' | 'local'): Storage | null {
  try {
    return kind === 'session' ? window.sessionStorage : window.localStorage
  } catch {
    return null // blocked (privacy settings, sandboxed frame)
  }
}

function read(kind: 'session' | 'local', provider: string): string | null {
  try {
    return storage(kind)?.getItem(PREFIX + provider) ?? null
  } catch {
    return null
  }
}

/** The user's key for `provider`, or null. */
export function getKey(provider: string | undefined): string | null {
  if (!provider) return null
  return read('session', provider) ?? read('local', provider) ?? memory.get(provider) ?? null
}

/** Whether the key for `provider` is kept on this device beyond the tab. */
export function isRemembered(provider: string): boolean {
  return read('local', provider) !== null
}

export function setKey(provider: string, key: string, remember: boolean): void {
  clearKey(provider)
  try {
    const target = storage(remember ? 'local' : 'session')
    if (target) {
      target.setItem(PREFIX + provider, key.trim())
      return
    }
  } catch {
    /* blocked or full: memory below */
  }
  memory.set(provider, key.trim())
}

export function clearKey(provider: string): void {
  memory.delete(provider)
  for (const kind of ['session', 'local'] as const) {
    try {
      storage(kind)?.removeItem(PREFIX + provider)
    } catch {
      /* blocked */
    }
  }
}

/** Which of `ids` have a key of the user's. */
export function keyedProviders(ids: string[]): string[] {
  return ids.filter((id) => getKey(id) !== null)
}

/** A shape check before saving; whether the key works is the provider's to say. */
export function looksLikeKey(key: string): boolean {
  const k = key.trim()
  return k.length >= 8 && k.length <= 512 && !/\s/.test(k)
}

/** The last four characters, enough to tell keys apart without showing one. */
export function maskKey(key: string): string {
  return `••••${key.slice(-4)}`
}

/**
 * The table as this user sees it: on a server that accepts user keys, a provider it has no
 * key for is available once the user has given theirs. The choice logic (assistantChoice.ts)
 * then works unchanged.
 */
export function withUserKeys(meta: AssistantMeta, keyed: string[]): AssistantMeta {
  if (!meta.byok) return meta
  const providers = meta.providers.map((p) =>
    !p.available && keyed.includes(p.id) ? { ...p, available: true } : p,
  )
  return { ...meta, available: providers.some((p) => p.available), providers }
}
