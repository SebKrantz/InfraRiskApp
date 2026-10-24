// Chat state and the stream-per-leg loop for the AI assistant.
//
// send() opens an SSE leg; when the loop suspends with `await_client`, the
// browser executes the UI tools against the bindings ref and posts the results
// back, which opens the next leg — until `done` ends the turn.

import { useCallback, useEffect, useRef, useState } from 'react'

import { pushAppState, streamChat, uploadAssistantFile } from '../lib/assistantApi'
import { executeClientTool, snapshot, type BindingsRef } from '../lib/assistantTools'
import type {
  AssistantArtifact,
  AssistantEvent,
  ClientToolOutcome,
  PendingToolCall,
  TranscriptItem,
} from '../types/assistant'

const STATE_PUSH_MS = 2500

export interface UseAssistant {
  items: TranscriptItem[]
  streaming: boolean
  conversationId: string | null
  send: (text: string, files: File[], provider?: string, model?: string) => void
  stop: () => void
  clear: () => void
}

export function useAssistant(bindings: BindingsRef, enabled: boolean): UseAssistant {
  const [items, setItems] = useState<TranscriptItem[]>([])
  const [streaming, setStreaming] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const conversationRef = useRef<string | null>(null)
  // provider/model choice lives with the panel; it passes them per send()
  const choiceRef = useRef<{ provider?: string; model?: string }>({})

  // Debounced app-state push, for external MCP clients. Idempotent under
  // StrictMode double-mount; skips identical snapshots.
  useEffect(() => {
    if (!enabled) return
    let last = ''
    const timer = setInterval(() => {
      const state = snapshot(bindings.current)
      const blob = JSON.stringify(state)
      if (blob !== last) {
        last = blob
        pushAppState(state)
      }
    }, STATE_PUSH_MS)
    return () => clearInterval(timer)
  }, [enabled, bindings])

  const append = useCallback((item: TranscriptItem) => {
    setItems((prev) => [...prev, item])
  }, [])

  const handleEvent = useCallback((e: AssistantEvent, pending: PendingToolCall[]) => {
    switch (e.type) {
      case 'start':
        conversationRef.current = e.conversation_id
        setConversationId(e.conversation_id)
        break
      case 'text':
        setItems((prev) => {
          const lastItem = prev[prev.length - 1]
          if (lastItem?.kind === 'text')
            return [...prev.slice(0, -1), { kind: 'text', text: lastItem.text + e.delta }]
          return [...prev, { kind: 'text', text: e.delta }]
        })
        break
      case 'thinking':
        setItems((prev) => {
          const lastItem = prev[prev.length - 1]
          if (lastItem?.kind === 'thinking')
            return [
              ...prev.slice(0, -1),
              { kind: 'thinking', text: lastItem.text + e.delta },
            ]
          return [...prev, { kind: 'thinking', text: e.delta }]
        })
        break
      case 'tool_call':
        setItems((prev) => [
          ...prev,
          {
            kind: 'tool',
            id: e.id,
            name: e.name,
            args: e.args,
            side: e.side,
            status: 'running',
            summary: '',
          },
        ])
        break
      case 'tool_result':
        setItems((prev) =>
          prev.map((it) =>
            it.kind === 'tool' && it.id === e.id
              ? { ...it, status: e.ok ? 'ok' : 'error', summary: e.summary }
              : it,
          ),
        )
        break
      case 'artifact':
        setItems((prev) => [
          ...prev,
          { kind: 'artifact', artifact: e as unknown as AssistantArtifact },
        ])
        break
      case 'await_client':
        pending.push(...e.calls)
        break
      case 'notice':
        setItems((prev) => [...prev, { kind: 'notice', message: e.message }])
        break
      case 'error':
        setItems((prev) => [...prev, { kind: 'error', message: e.message }])
        break
      case 'usage':
      case 'done':
        break
    }
  }, [])

  const runLeg = useCallback(
    async (body: {
      message?: string
      tool_results?: ClientToolOutcome[]
      files?: string[]
    }): Promise<void> => {
      const ctl = abortRef.current ?? new AbortController()
      abortRef.current = ctl
      const pending: PendingToolCall[] = []
      await streamChat(
        {
          conversation_id: conversationRef.current ?? undefined,
          ...choiceRef.current,
          ...body,
          app_state: snapshot(bindings.current),
        },
        (e) => handleEvent(e, pending),
        ctl.signal,
      )
      if (ctl.signal.aborted) return
      if (pending.length > 0) {
        const results: ClientToolOutcome[] = []
        for (const call of pending) {
          const outcome = await executeClientTool(bindings, call, conversationRef.current)
          setItems((prev) =>
            prev.map((it) =>
              it.kind === 'tool' && it.id === call.id
                ? {
                    ...it,
                    status: outcome.ok ? 'ok' : 'error',
                    summary: outcome.ok
                      ? String(outcome.result ?? '')
                      : (outcome.error ?? ''),
                  }
                : it,
            ),
          )
          results.push(outcome)
          if (ctl.signal.aborted) return
        }
        await runLeg({ tool_results: results })
      }
    },
    [bindings, handleEvent],
  )

  const send = useCallback(
    (text: string, files: File[], provider?: string, model?: string) => {
      if (streaming || (!text.trim() && files.length === 0)) return
      choiceRef.current = { provider, model }
      setStreaming(true)
      abortRef.current = new AbortController()
      void (async () => {
        try {
          const names: string[] = []
          if (files.length > 0) {
            // uploads need a conversation id; mint one if none exists yet
            const cid =
              conversationRef.current ??
              (conversationRef.current = crypto.randomUUID().replace(/-/g, ''))
            setConversationId(cid)
            for (const f of files) {
              const up = await uploadAssistantFile(cid, f)
              names.push(up.name)
            }
          }
          append({ kind: 'user', text, files: names })
          await runLeg({ message: text, files: names.length ? names : undefined })
        } catch (e) {
          if ((e as Error).name !== 'AbortError')
            append({ kind: 'error', message: (e as Error).message })
        } finally {
          setStreaming(false)
          abortRef.current = null
        }
      })()
    },
    [append, runLeg, streaming],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setStreaming(false)
  }, [])

  const clear = useCallback(() => {
    stop()
    const cid = conversationRef.current
    conversationRef.current = null
    setConversationId(null)
    setItems([])
    if (cid) {
      void fetch(`/api/assistant/conversations/${encodeURIComponent(cid)}`, {
        method: 'DELETE',
      }).catch(() => undefined)
    }
  }, [stop])

  return { items, streaming, conversationId, send, stop, clear }
}
