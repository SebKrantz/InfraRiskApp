// The assistant's mount point: FAB when closed, panel when open. Rendered from
// App.tsx so the client tools reach every setter through the bindings ref with
// no extra prop drilling.
//
// It renders nothing at all until /api/assistant/meta says a provider is
// configured, so an app without API keys looks exactly as it did before.

import { useEffect, useState } from 'react'

import { getAssistantMeta } from '../../lib/assistantApi'
import {
  HANDOFF,
  applyHandoff,
  loadChoice,
  resolveChoice,
  saveChoice,
} from '../../lib/assistantChoice'
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

  // The saved choice, checked against the table; an AGUI handoff (ai_* parameters, read at
  // module load) replaces it and is saved like a pick of the user's.
  useEffect(() => {
    let live = true
    getAssistantMeta()
      .then((m) => {
        if (!live || !m.available) return
        let next = resolveChoice(m, loadChoice())
        const handed = HANDOFF && applyHandoff(m, HANDOFF, next)
        if (handed) {
          next = handed
          saveChoice(handed)
        }
        setMeta(m)
        setChoice(next)
      })
      .catch(() => undefined) // no assistant configured: stay invisible
    return () => {
      live = false
    }
  }, [])

  const assistant = useAssistant(bindings, !!meta?.available)
  if (!meta?.available || !choice) return null
  const pick = (c: AssistantChoice) => {
    setChoice(c)
    saveChoice(c)
  }
  return open ? (
    <AssistantPanel
      assistant={assistant}
      meta={meta}
      choice={choice}
      onChoice={pick}
      onClose={() => setOpen(false)}
    />
  ) : (
    <AssistantFab onClick={() => setOpen(true)} />
  )
}
