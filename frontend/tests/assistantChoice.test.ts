// Choice validation and the AGUI handoff (src/lib/assistantChoice.ts). No test runner in
// this repo: Node runs it directly (type stripping), from frontend/:  node --test tests/
import assert from 'node:assert/strict'
import { test } from 'node:test'

// A browser stand-in, installed before the module loads: its HANDOFF is read at import.
const calls: string[] = []
const storage = new Map<string, string>()
const location = {
  href: 'http://localhost:9360/?assistant=1&ai_provider=gemini&ai_model=gemini-3.1-pro-preview&ai_effort=high&ai_tier=flex&x=2#map',
  search: '?assistant=1&ai_provider=gemini&ai_model=gemini-3.1-pro-preview&ai_effort=high&ai_tier=flex&x=2',
}
Object.assign(globalThis, {
  window: {
    location,
    history: { state: null, replaceState: (_s: unknown, _t: string, url: URL) => calls.push(String(url)) },
    localStorage: {
      getItem: (k: string) => storage.get(k) ?? null,
      setItem: (k: string, v: string) => void storage.set(k, v),
    },
  },
})

const {
  HANDOFF, applyHandoff, loadChoice, parseHandoff, resolveChoice, saveChoice, withModel, withProvider,
} = await import('../src/lib/assistantChoice.ts')

const five = ['low', 'medium', 'high', 'xhigh', 'max']
// 3.8 Flash and 3.1 Pro answer `minimal` with a 400; only Flash-Lite takes it
const three = ['low', 'medium', 'high']
const provider = (id: string, label: string, available: boolean, models: string[], def: string,
  efforts: Record<string, string[]>, tiers: string[]) => ({
  id, label, available, models, default_model: def, efforts,
  effort_default: Object.fromEntries(models.map((m) => [m, efforts[m].length ? 'high' : null])),
  tiers,
})
const meta = (keyed: string[] = ['anthropic', 'gemini', 'openai']) => ({
  available: true,
  default_provider: keyed[0] ?? null,
  providers: [
    provider('anthropic', 'Claude', keyed.includes('anthropic'),
      ['claude-opus-5-5', 'claude-fable-5-1', 'claude-sonnet-5-5', 'claude-haiku-4-5'], 'claude-sonnet-5-5',
      { 'claude-opus-5-5': five, 'claude-fable-5-1': five, 'claude-sonnet-5-5': five, 'claude-haiku-4-5': [] },
      ['standard']),
    provider('gemini', 'Gemini', keyed.includes('gemini'),
      ['gemini-3.8-flash', 'gemini-3.1-pro-preview', 'gemini-3.5-flash-lite'], 'gemini-3.8-flash',
      { 'gemini-3.8-flash': three, 'gemini-3.1-pro-preview': three,
        'gemini-3.5-flash-lite': ['minimal', ...three] },
      ['standard', 'flex']),
    provider('openai', 'OpenAI', keyed.includes('openai'),
      ['gpt-6-luna', 'gpt-6.1-sol', 'gpt-6-astra'], 'gpt-6-luna',
      { 'gpt-6-luna': five, 'gpt-6.1-sol': five, 'gpt-6-astra': five }, ['standard', 'flex']),
  ],
})
const none = { provider: null, model: null, effort: null, tier: null }
const saved = { provider: 'openai', model: 'gpt-6.1-sol', effort: 'max', tier: 'flex' } as const

test('the ai_* parameters are taken off the address bar, the rest kept', () => {
  assert.deepEqual(HANDOFF, {
    provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: 'high', tier: 'flex',
  })
  assert.deepEqual(calls, ['http://localhost:9360/?assistant=1&x=2#map'])
})

test('parseHandoff', () => {
  assert.equal(parseHandoff('?assistant=1'), null)
  assert.deepEqual(parseHandoff('?ai_provider=openai&ai_tier=flex'), { ...none, provider: 'openai', tier: 'flex' })
})

test('resolveChoice: default, saved, and anything no longer on offer', () => {
  const m = meta()
  assert.deepEqual(resolveChoice(m, null),
    { provider: 'anthropic', model: 'claude-sonnet-5-5', effort: null, tier: 'standard' })
  assert.deepEqual(resolveChoice(m, saved),
    { provider: 'openai', model: 'gpt-6.1-sol', effort: 'max', tier: 'flex' })
  // an old model ID, an effort the model does not take, flex on Claude
  assert.deepEqual(resolveChoice(m, { provider: 'anthropic', model: 'claude-sonnet-5', effort: 'max', tier: 'flex' }),
    { provider: 'anthropic', model: 'claude-sonnet-5-5', effort: null, tier: 'standard' })
  assert.deepEqual(resolveChoice(m, { provider: 'anthropic', model: 'claude-haiku-4-5', effort: 'high', tier: 'standard' }),
    { provider: 'anthropic', model: 'claude-haiku-4-5', effort: null, tier: 'standard' })
  // an unkeyed provider falls back to the default provider
  assert.deepEqual(resolveChoice(meta(['gemini']), saved),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'standard' })
  // a missing provider likewise: nothing carried over
  assert.deepEqual(resolveChoice(m, { model: 'gpt-6.1-sol', effort: 'max', tier: 'flex' }),
    { provider: 'anthropic', model: 'claude-sonnet-5-5', effort: null, tier: 'standard' })
  // flex where the provider has none drops to Standard, the valid effort stays
  assert.deepEqual(resolveChoice(m, { provider: 'anthropic', model: 'claude-opus-5-5', effort: 'high', tier: 'flex' }),
    { provider: 'anthropic', model: 'claude-opus-5-5', effort: 'high', tier: 'standard' })
  // an unlisted saved model: the provider's default model and Default effort (flex is the provider's, kept)
  assert.deepEqual(resolveChoice(m, { provider: 'gemini', model: 'gemini-3-flash-preview', effort: 'low', tier: 'flex' }),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'flex' })
  // `minimal` saved under the old table is not a 3.8 Flash level; 3.1 Pro's medium now is
  assert.equal(resolveChoice(m, { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'minimal', tier: 'standard' }).effort, null)
  assert.equal(resolveChoice(m, { provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: 'medium', tier: 'standard' }).effort, 'medium')
})

test('saved choice round-trips through localStorage', () => {
  const c = { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'low', tier: 'flex' } as const
  saveChoice(c)
  assert.deepEqual(loadChoice(), c)
})

test('changing provider and model', () => {
  const m = meta()
  const cur = { provider: 'openai', model: 'gpt-6.1-sol', effort: 'max', tier: 'flex' } as const
  assert.deepEqual(withProvider(m, cur, 'gemini'),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'flex' })
  assert.deepEqual(withProvider(m, cur, 'anthropic'),
    { provider: 'anthropic', model: 'claude-sonnet-5-5', effort: null, tier: 'standard' })
  assert.deepEqual(withModel(m, cur, 'gpt-6-astra'), { ...cur, model: 'gpt-6-astra' })
  const gem = { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'medium', tier: 'standard' } as const
  assert.equal(withModel(m, gem, 'gemini-3.1-pro-preview').effort, 'medium') // 3.1 Pro takes medium
  assert.equal(withModel(m, gem, 'gemini-3.5-flash-lite').effort, 'medium')
  const lite = { ...gem, model: 'gemini-3.5-flash-lite', effort: 'minimal' } as const
  assert.equal(withModel(m, lite, 'gemini-3.8-flash').effort, null) // only Flash-Lite takes minimal
  assert.equal(withModel(m, lite, 'gemini-3.1-pro-preview').effort, null)
})

test('handoff is a complete setting: saved effort and tier never survive', () => {
  const m = meta()
  const current = resolveChoice(m, saved) // openai / sol / max / flex
  // provider + model only: Default effort, Standard
  assert.deepEqual(applyHandoff(m, { ...none, provider: 'anthropic', model: 'claude-opus-5-5' }, current),
    { provider: 'anthropic', model: 'claude-opus-5-5', effort: null, tier: 'standard' })
  // with effort and flex
  assert.deepEqual(
    applyHandoff(m, { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'low', tier: 'flex' }, current),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'low', tier: 'flex' })
  assert.deepEqual(
    applyHandoff(m, { provider: 'gemini', model: 'gemini-3.5-flash-lite', effort: 'minimal', tier: 'flex' }, current),
    { provider: 'gemini', model: 'gemini-3.5-flash-lite', effort: 'minimal', tier: 'flex' })
  // an effort the model does not take and flex the provider lacks are dropped
  assert.deepEqual(
    applyHandoff(m, { provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: 'minimal', tier: 'standard' }, current),
    { provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: null, tier: 'standard' })
  assert.deepEqual(
    applyHandoff(m, { provider: 'anthropic', model: 'claude-haiku-4-5', effort: 'high', tier: 'flex' }, current),
    { provider: 'anthropic', model: 'claude-haiku-4-5', effort: null, tier: 'standard' })
  // a provider alone: its default model
  assert.deepEqual(applyHandoff(m, { ...none, provider: 'gemini' }, current),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'standard' })
  // a model alone names its provider
  assert.deepEqual(applyHandoff(m, { ...none, model: 'gpt-6-astra' }, current),
    { provider: 'openai', model: 'gpt-6-astra', effort: null, tier: 'standard' })
  // a model of another provider than the one named: the model is dropped for that provider's default
  assert.deepEqual(applyHandoff(m, { ...none, provider: 'gemini', model: 'gpt-6-luna' }, current),
    { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'standard' })
})

test('handoff: an unkeyed provider or unknown model is ignored', () => {
  const m = meta(['anthropic'])
  const current = resolveChoice(m, null)
  assert.equal(applyHandoff(m, { ...none, provider: 'openai', model: 'gpt-6-luna' }, current), null)
  assert.equal(applyHandoff(m, { ...none, provider: 'nope', model: 'nope' }, current), null)
  // an unusable provider, a usable model: the model's provider is used
  assert.deepEqual(applyHandoff(m, { ...none, provider: 'openai', model: 'claude-opus-5-5', effort: 'low' }, current),
    { provider: 'anthropic', model: 'claude-opus-5-5', effort: 'low', tier: 'standard' })
})

test('handoff with only ai_effort / ai_tier changes just those, when valid', () => {
  const m = meta()
  const current = { provider: 'gemini', model: 'gemini-3.8-flash', effort: 'low', tier: 'standard' } as const
  assert.deepEqual(applyHandoff(m, { ...none, effort: 'high', tier: 'flex' }, current),
    { ...current, effort: 'high', tier: 'flex' })
  assert.equal(applyHandoff(m, { ...none, effort: 'max' }, current), null) // not a level of this model
  assert.equal(applyHandoff(m, { ...none, tier: 'flex' }, { ...current, provider: 'anthropic', model: 'claude-opus-5-5' }), null)
})

// ---- review-fixes-spec H: the five cases, end to end ------------------------------------
// Each case loads a fresh copy of the module (its HANDOFF is read from the address bar at
// import), then does what Assistant.tsx does when /meta arrives.
let boots = 0
async function boot(url: string, savedChoice: Record<string, string | null> | null = null) {
  const u = new URL(url)
  const replaced: string[] = []
  Object.assign(globalThis, {
    window: {
      location: { href: u.href, search: u.search },
      history: { state: null, replaceState: (_s: unknown, _t: string, to: URL) => replaced.push(String(to)) },
      localStorage: { getItem: () => null, setItem: () => undefined },
    },
  })
  const mod = await import(`../src/lib/assistantChoice.ts?boot=${++boots}`)
  const m = meta()
  let choice = mod.resolveChoice(m, savedChoice)
  const handed = mod.HANDOFF && mod.applyHandoff(m, mod.HANDOFF, choice)
  if (handed) choice = handed
  return { choice, bar: replaced.at(-1) ?? null, replaced }
}
const at = 'http://127.0.0.1:8001/'

test('H1: provider, model, effort and flex; the bar keeps ?assistant=1', async () => {
  const r = await boot(`${at}?assistant=1&ai_provider=openai&ai_model=gpt-6.1-sol&ai_effort=xhigh&ai_tier=flex`)
  assert.deepEqual(r.choice, { provider: 'openai', model: 'gpt-6.1-sol', effort: 'xhigh', tier: 'flex' })
  assert.equal(r.bar, `${at}?assistant=1`)
})

test('H2: a handoff is a complete setting; what was saved does not survive', async () => {
  const r = await boot(`${at}?assistant=1&ai_provider=anthropic&ai_model=claude-sonnet-5-5`,
    { provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: 'low', tier: 'flex' })
  assert.deepEqual(r.choice, { provider: 'anthropic', model: 'claude-sonnet-5-5', effort: null, tier: 'standard' })
  assert.equal(r.bar, `${at}?assistant=1`)
})

test('H3: a model alone names its provider', async () => {
  const r = await boot(`${at}?ai_model=gpt-6-luna`)
  assert.deepEqual(r.choice, { provider: 'openai', model: 'gpt-6-luna', effort: null, tier: 'standard' })
  assert.equal(r.bar, at)
})

test('H4: minimal is not a 3.8 Flash level', async () => {
  const r = await boot(`${at}?ai_provider=gemini&ai_model=gemini-3.8-flash&ai_effort=minimal`)
  assert.deepEqual(r.choice, { provider: 'gemini', model: 'gemini-3.8-flash', effort: null, tier: 'standard' })
})

test('H5: other parameters and the hash stay in the bar; 3.1 Pro takes medium', async () => {
  const r = await boot(`${at}?x=1&ai_provider=gemini&ai_model=gemini-3.1-pro-preview&ai_effort=medium#h`)
  assert.deepEqual(r.choice, { provider: 'gemini', model: 'gemini-3.1-pro-preview', effort: 'medium', tier: 'standard' })
  assert.equal(r.bar, `${at}?x=1#h`)
})

test('H: no ai_* parameters leaves the address bar alone', async () => {
  const r = await boot(`${at}?assistant=1`)
  assert.deepEqual(r.replaced, [])
})
