// The assistant's mount point: FAB when closed, panel when open. Rendered from
// App.tsx so the client tools reach every setter through the bindings ref with
// no extra prop drilling.
//
// It renders nothing at all until /api/assistant/meta says a provider is
// configured, so an app without API keys looks exactly as it did before.

import { useEffect, useState } from 'react'

import { getAssistantMeta } from '../../lib/assistantApi'
import { useAssistant } from '../../hooks/useAssistant'
import type { BindingsRef } from '../../lib/assistantTools'
import type { AssistantMeta } from '../../types/assistant'
import AssistantFab from './AssistantFab'
import AssistantPanel from './AssistantPanel'

export default function Assistant({ bindings }: { bindings: BindingsRef }) {
  const [meta, setMeta] = useState<AssistantMeta | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let live = true
    getAssistantMeta()
      .then((m) => {
        if (live) setMeta(m.available ? m : null)
      })
      .catch(() => undefined) // no assistant configured: stay invisible
    return () => {
      live = false
    }
  }, [])

  const assistant = useAssistant(bindings, !!meta?.available)
  if (!meta?.available) return null
  return open ? (
    <AssistantPanel assistant={assistant} meta={meta} onClose={() => setOpen(false)} />
  ) : (
    <AssistantFab onClick={() => setOpen(true)} />
  )
}
