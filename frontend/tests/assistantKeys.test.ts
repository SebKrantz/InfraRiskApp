// Bring your own key (src/lib/assistantKeys.ts): where a user's key is kept in the browser,
// and how it makes a provider available. No test runner in this repo: Node runs it directly
// (type stripping), from frontend/:  node --test tests/*.test.ts
import assert from 'node:assert/strict'
import { test } from 'node:test'

// A browser stand-in with both storages, installed before the modules load.
const fakeStorage = () => {
  const m = new Map<string, string>()
  return {
    map: m,
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
  }
}
const session = fakeStorage()
const local = fakeStorage()
let blocked = false
const win = {
  location: { href: 'http://localhost:9360/', search: '' },
  history: { state: null, replaceState: () => undefined },
  get sessionStorage() {
    if (blocked) throw new Error('SecurityError')
    return session
  },
  get localStorage() {
    if (blocked) throw new Error('SecurityError')
    return local
  },
}
Object.assign(globalThis, { window: win })

const keys = await import('../src/lib/assistantKeys.ts')
const { resolveChoice } = await import('../src/lib/assistantChoice.ts')

const KEY = 'sk-ant-api03-testkey-0123456789'
const provider = (id: string, available: boolean) => ({
  id, label: id, available, models: [`${id}-model`], default_model: `${id}-model`,
  efforts: { [`${id}-model`]: [] }, effort_default: { [`${id}-model`]: null }, tiers: ['standard'],
})
const meta = (byok: boolean) => ({
  available: false,
  byok,
  default_provider: null,
  providers: [provider('anthropic', false), provider('gemini', false), provider('openai', false)],
})

test('a key stays in this tab unless remembered', () => {
  keys.setKey('anthropic', `  ${KEY}  `, false)
  assert.equal(keys.getKey('anthropic'), KEY)
  assert.equal(keys.isRemembered('anthropic'), false)
  assert.equal(local.map.size, 0)
  keys.setKey('anthropic', KEY, true)
  assert.equal(keys.isRemembered('anthropic'), true)
  assert.equal(session.map.size, 0, 'remembering moves it, it does not copy it')
  keys.clearKey('anthropic')
  assert.equal(keys.getKey('anthropic'), null)
  assert.equal(session.map.size + local.map.size, 0)
})

test('with storage blocked it lives in memory only', () => {
  blocked = true
  try {
    keys.setKey('openai', KEY, true)
    assert.equal(keys.getKey('openai'), KEY)
    keys.clearKey('openai')
    assert.equal(keys.getKey('openai'), null)
  } finally {
    blocked = false
  }
})

test('a user key makes its provider available, only on a server that takes them', () => {
  keys.setKey('gemini', KEY, false)
  const keyed = keys.keyedProviders(['anthropic', 'gemini', 'openai'])
  assert.deepEqual(keyed, ['gemini'])
  assert.deepEqual(keys.withUserKeys(meta(false), keyed), meta(false))
  const seen = keys.withUserKeys(meta(true), keyed)
  assert.equal(seen.available, true)
  assert.deepEqual(seen.providers.filter((p) => p.available).map((p) => p.id), ['gemini'])
  // the choice logic then lands on it, with no server key anywhere
  assert.equal(resolveChoice(seen, null).provider, 'gemini')
  keys.clearKey('gemini')
})

test('shape check and mask', () => {
  assert.equal(keys.looksLikeKey(KEY), true)
  assert.equal(keys.looksLikeKey('short'), false)
  assert.equal(keys.looksLikeKey('has a space inside it'), false)
  assert.equal(keys.maskKey(KEY), '••••6789')
})
