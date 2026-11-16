// The assistant's "Models & execution" popover, mirroring AGUI's: provider, model, reasoning
// effort, and the Standard | Flex execution control. The trigger reads
// "<Provider> · <model>", plus " · <effort>" when not the default and " · Flex" when flex.
//
// The popover is positioned against the nearest positioned ancestor (the panel header), so
// it spans the panel's width and the notes below fit on two lines.
//
// On a server that accepts user keys (meta.byok) it also holds "Your API keys": one row per
// provider to paste, see (masked) or remove the user's own key (lib/assistantKeys.ts).

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

import { canFlex, effortsFor, withModel, withProvider } from '../../lib/assistantChoice'
import {
  KEY_HINTS,
  clearKey,
  getKey,
  isRemembered,
  looksLikeKey,
  maskKey,
  setKey,
} from '../../lib/assistantKeys'
import type { AssistantChoice, AssistantMeta, AssistantProvider } from '../../types/assistant'

const FLEX_NOTE =
  'Flex: about half the price, but slower (minutes per step); falls back to standard when busy.'

const KEY_NOTE =
  'Your key stays in this browser (this tab only, unless remembered). Each request sends it ' +
  "to this app's server, which passes it to the provider for that request and never stores or logs it."

const selectCls =
  'min-w-0 flex-1 rounded border border-gray-700 bg-gray-800 px-2 py-1 font-mono text-[11.5px] text-gray-200 focus:border-blue-500 focus:outline-none'

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-t border-gray-800 pb-1">
      <div className="px-3 pb-0.5 pt-2 text-[10px] font-medium uppercase tracking-widest text-gray-500">
        {title}
      </div>
      {children}
    </div>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center gap-2 px-3 py-1">
      <span className="w-16 shrink-0 text-[10px] font-medium uppercase tracking-widest text-gray-500">
        {label}
      </span>
      {children}
    </label>
  )
}

function KeyRow({
  provider,
  remember,
  onKeys,
}: {
  provider: AssistantProvider
  remember: boolean
  onKeys: (provider: string) => void
}) {
  const [draft, setDraft] = useState('')
  const key = getKey(provider.id)
  const save = () => {
    if (!looksLikeKey(draft)) return
    setKey(provider.id, draft, remember)
    setDraft('')
    onKeys(provider.id)
  }
  return (
    <div className="flex items-center gap-2 px-3 py-1">
      <span className="w-16 shrink-0 truncate text-[11px] text-gray-300">{provider.label}</span>
      {key ? (
        <>
          <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-emerald-300">
            {maskKey(key)}{' '}
            <span className="font-sans text-[10px] text-gray-500">
              {isRemembered(provider.id) ? 'remembered' : 'this tab'}
            </span>
          </span>
          <button
            type="button"
            onClick={() => {
              clearKey(provider.id)
              onKeys(provider.id)
            }}
            className="shrink-0 rounded px-1.5 py-0.5 text-[11px] text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          >
            Remove
          </button>
        </>
      ) : (
        <>
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') save()
            }}
            autoComplete="off"
            spellCheck={false}
            aria-label={`${provider.label} API key`}
            placeholder={`${KEY_HINTS[provider.id] ?? 'API key'}${provider.available ? ' (optional)' : ''}`}
            className={selectCls}
          />
          <button
            type="button"
            onClick={save}
            disabled={!looksLikeKey(draft)}
            className="shrink-0 rounded bg-blue-600 px-1.5 py-0.5 text-[11px] text-white hover:bg-blue-500 disabled:opacity-40"
          >
            Save
          </button>
        </>
      )}
    </div>
  )
}

export default function ModelPicker({
  meta,
  choice,
  onChange,
  onKeys,
  disabled,
  openOnMount,
}: {
  meta: AssistantMeta
  choice: AssistantChoice
  onChange: (choice: AssistantChoice) => void
  /** a key of the user's for `provider` was added or removed */
  onKeys: (provider: string) => void
  disabled?: boolean
  /** open at once (no provider has a key yet: the key rows are where to start) */
  openOnMount?: boolean
}) {
  const [open, setOpen] = useState(!!openOnMount)
  const [remember, setRemember] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const provider = meta.providers.find((p) => p.id === choice.provider)
  const levels = effortsFor(provider, choice.model)
  const flexOffered = canFlex(provider)
  const flex = flexOffered && choice.tier === 'flex'
  const fallback = provider?.effort_default?.[choice.model] ?? null
  const sep = <span className="text-gray-600">·</span>

  return (
    <div ref={box} className="contents">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title="Models & execution"
        aria-label="Models & execution"
        className="flex min-w-0 flex-1 items-center gap-1.5 rounded-md border border-gray-700 bg-gray-800 px-2 py-1 text-[11px] text-gray-300 hover:border-gray-600 disabled:opacity-50"
      >
        <span className="shrink-0">{provider?.label ?? choice.provider}</span>
        {sep}
        <span className="truncate font-mono">{choice.model}</span>
        {choice.effort && (
          <>
            {sep}
            <span className="shrink-0 font-mono">{choice.effort}</span>
          </>
        )}
        {flex && (
          <>
            {sep}
            <span className="shrink-0 text-amber-400">Flex</span>
          </>
        )}
        <ChevronDown className="ml-auto h-3 w-3 shrink-0 text-gray-500" />
      </button>
      {open && (
        <div className="absolute inset-x-2 top-full z-40 mt-1 max-h-[calc(100vh-4rem)] overflow-y-auto rounded-md border border-gray-700 bg-gray-900 py-1 shadow-xl shadow-black/50">
          <div className="pb-1 pt-1">
            <Field label="Provider">
              <select
                value={choice.provider}
                onChange={(e) => onChange(withProvider(meta, choice, e.target.value))}
                className={selectCls}
              >
                {meta.providers.map((p) => (
                  <option key={p.id} value={p.id} disabled={!p.available}>
                    {p.available ? p.label : `${p.label} (${meta.byok ? 'add a key below' : 'no key'})`}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          {meta.byok && (
            <Section title="Your API keys">
              {meta.providers.map((p) => (
                <KeyRow key={p.id} provider={p} remember={remember} onKeys={onKeys} />
              ))}
              <label className="flex items-center gap-2 px-3 py-1 text-[11px] text-gray-400">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                  className="h-3 w-3 accent-blue-500"
                />
                Remember new keys on this device
              </label>
              <div className="px-3 pb-1 text-[11px] leading-snug text-gray-500">{KEY_NOTE}</div>
            </Section>
          )}

          <Section title="Assistant">
            <Field label="Model">
              <select
                value={choice.model}
                onChange={(e) => onChange(withModel(meta, choice, e.target.value))}
                className={selectCls}
              >
                {provider?.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </Field>
            {levels.length > 0 && (
              <Field label="Effort">
                <select
                  value={choice.effort ?? ''}
                  onChange={(e) => onChange({ ...choice, effort: e.target.value || null })}
                  className={selectCls}
                >
                  <option value="">{fallback ? `Default (${fallback})` : 'Default'}</option>
                  {levels.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
              </Field>
            )}
          </Section>

          <Section title="Execution">
            <div className="px-3 pt-1">
              <div
                role="radiogroup"
                aria-label="Execution mode"
                className="flex overflow-hidden rounded-md border border-gray-700"
              >
                {(['standard', 'flex'] as const).map((tier) => {
                  const on = (tier === 'flex') === flex
                  const off = tier === 'flex' && !flexOffered
                  return (
                    <button
                      key={tier}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      disabled={off}
                      onClick={() => {
                        if (!on) onChange({ ...choice, tier })
                      }}
                      className={`flex-1 px-2 py-1 text-[12px] capitalize transition-colors ${
                        tier === 'flex' ? 'border-l border-gray-700' : ''
                      } ${
                        on
                          ? 'bg-blue-500/15 text-blue-300'
                          : off
                            ? 'cursor-not-allowed text-gray-600'
                            : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                      }`}
                    >
                      {tier}
                    </button>
                  )
                })}
              </div>
              <div className="pb-1 pt-1.5 text-[11px] leading-snug text-gray-500">
                {flexOffered ? FLEX_NOTE : `${provider?.label ?? 'This provider'} has no flex tier.`}
              </div>
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
