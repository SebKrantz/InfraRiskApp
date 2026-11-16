// The assistant's mount point: FAB when closed, panel when open. Rendered from
// App.tsx so the client tools reach every setter through the bindings ref with
// no extra prop drilling.
//
// It renders nothing at all until /api/assistant/meta says a provider is
// configured, or that users may bring their own key (lib/assistantKeys.ts), so an
// app without API keys or user keys looks exactly as it did before.

import { useEffect, useMemo, useState } from 'react'

import { getAssistantMeta } from '../../lib/assistantApi'
import {
  HANDOFF,
  applyHandoff,
  loadChoice,
  resolveChoice,
  saveChoice,
} from '../../lib/assistantChoice'
import { keyedProviders, withUserKeys } from '../../lib/assistantKeys'
import { useAssistant } from '../../hooks/useAssistant'
import type { BindingsRef } from '../../lib/assistantTools'
import type { AssistantChoice, AssistantMeta } from '../../types/assistant'
import AssistantFab from './AssistantFab'
import AssistantPanel from './AssistantPanel'

// `?assistant=1` opens the panel on load (AGUI's agent links use it). Read once at module
// load, before the app can rewrite location.search to mirror its own state.
const OPEN_ON_LOAD = new URLSearchParams(window.location.search).get('assistant') === '1'

export default function Assistant({ bindings }: { bindings: BindingsRef }) {
  const [meta, setMeta] = useState<AssistantMeta | null>(null)
  const [choice, setChoice] = useState<AssistantChoice | null>(null)
  const [open, setOpen] = useState(OPEN_ON_LOAD)
  // providers the user has given a key for; the table below counts them as available
  const [keyed, setKeyed] = useState<string[]>([])
  const view = useMemo(() => (meta ? withUserKeys(meta, keyed) : null), [meta, keyed])

  // The saved choice, checked against the table; an AGUI handoff (ai_* parameters, read at
  // module load) replaces it and is saved like a pick of the user's.
  useEffect(() => {
    let live = true
    getAssistantMeta()
      .then((m) => {
        if (!live || !(m.available || m.byok)) return
        const own = keyedProviders(m.providers.map((p) => p.id))
        const seen = withUserKeys(m, own)
        let next = resolveChoice(seen, loadChoice())
        const handed = HANDOFF && applyHandoff(seen, HANDOFF, next)
        if (handed) {
          next = handed
          saveChoice(handed)
        }
        setMeta(m)
        setKeyed(own)
        setChoice(next)
      })
      .catch(() => undefined) // no assistant configured: stay invisible
    return () => {
      live = false
    }
  }, [])

  const assistant = useAssistant(bindings, !!view)
  if (!view || !choice) return null
  const pick = (c: AssistantChoice) => {
    setChoice(c)
    saveChoice(c)
  }
  // a key was added or removed: the choice moves to a provider that is still available
  // when its own lost its key, and to the provider just keyed when there was none before
  const onKeys = (provider: string) => {
    const now = keyedProviders(view.providers.map((p) => p.id))
    setKeyed(now)
    const next = withUserKeys(meta as AssistantMeta, now)
    const current = next.providers.find((p) => p.id === choice.provider)
    const target = current?.available ? choice : { ...choice, provider }
    const resolved = resolveChoice(next, target)
    setChoice(resolved)
    saveChoice(resolved)
  }
  return open ? (
    <AssistantPanel
      assistant={assistant}
      meta={view}
      choice={choice}
      onChoice={pick}
      onKeys={onKeys}
      onClose={() => setOpen(false)}
    />
  ) : (
    <AssistantFab onClick={() => setOpen(true)} />
  )
}
