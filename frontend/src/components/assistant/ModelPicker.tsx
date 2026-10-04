// The assistant's "Models & execution" popover, mirroring AGUI's: provider, model, reasoning
// effort, and the Standard | Flex execution control. The trigger reads
// "<Provider> · <model>", plus " · <effort>" when not the default and " · Flex" when flex.
//
// The popover is positioned against the nearest positioned ancestor (the panel header), so
// it spans the panel's width and the notes below fit on two lines.

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

import { canFlex, effortsFor, withModel, withProvider } from '../../lib/assistantChoice'
import type { AssistantChoice, AssistantMeta } from '../../types/assistant'

const FLEX_NOTE =
  'Flex: about half the price, but slower (minutes per step); falls back to standard when busy.'

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

export default function ModelPicker({
  meta,
  choice,
  onChange,
  disabled,
}: {
  meta: AssistantMeta
  choice: AssistantChoice
  onChange: (choice: AssistantChoice) => void
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
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
                    {p.available ? p.label : `${p.label} (no key)`}
                  </option>
                ))}
              </select>
            </Field>
          </div>

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
