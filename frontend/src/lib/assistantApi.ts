// Chat transport for the AI assistant: SSE over POST, raw-body uploads.
//
// EventSource cannot POST, so the stream is read by hand off fetch's
// ReadableStream: `event: <type>` / `data: <json>` pairs separated by blank
// lines, comment lines (`: ping` heartbeats) skipped.

import type { AssistantEvent, AssistantMeta, ClientToolOutcome } from '../types/assistant'

export interface ChatRequest {
  conversation_id?: string
  provider?: string
  model?: string
  message?: string
  tool_results?: ClientToolOutcome[]
  files?: string[]
  app_state?: Record<string, unknown>
}

export async function getAssistantMeta(): Promise<AssistantMeta> {
  const res = await fetch('/api/assistant/meta')
  if (!res.ok) throw new Error(`assistant unavailable (${res.status})`)
  return res.json()
}

export async function streamChat(
  body: ChatRequest,
  onEvent: (e: AssistantEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/assistant/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    let detail = `${res.status}`
    try {
      const data = (await res.json()) as { detail?: string }
      if (data.detail) detail = data.detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE frames are separated by a blank line.
    for (;;) {
      const cut = buffer.indexOf('\n\n')
      if (cut < 0) break
      const frame = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith(':')) continue // heartbeat
        if (line.startsWith('event: ')) event = line.slice(7).trim()
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (!data) continue
      try {
        onEvent({ type: event, ...JSON.parse(data) } as AssistantEvent)
      } catch {
        // malformed frame — skip rather than kill the stream
      }
    }
  }
}

export async function uploadAssistantFile(
  conversationId: string,
  file: File,
): Promise<{ conversation_id: string; name: string; bytes: number }> {
  const res = await fetch(
    `/api/assistant/conversations/${encodeURIComponent(conversationId)}/files?filename=${encodeURIComponent(file.name)}`,
    { method: 'POST', body: file },
  )
  const data = (await res.json()) as {
    detail?: string
    conversation_id: string
    name: string
    bytes: number
  }
  if (!res.ok) throw new Error(data.detail ?? `upload failed (${res.status})`)
  return data
}

export function pushAppState(state: Record<string, unknown>): void {
  // Fire-and-forget; the snapshot is advisory context for MCP clients.
  void fetch('/api/assistant/app_state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
  }).catch(() => undefined)
}
