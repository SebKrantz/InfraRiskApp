// Types for the AI assistant: availability, SSE events, transcript, client tools.

export interface AssistantProvider {
  id: string
  label: string
  models: string[]
  default_model: string
  available: boolean
}

export interface AssistantMeta {
  available: boolean
  default_provider: string | null
  providers: AssistantProvider[]
}

export interface AssistantArtifact {
  id: string
  kind: 'chart' | 'map' | 'png' | 'docx' | 'xlsx' | 'csv' | 'gpkg' | 'file'
  filename: string
  title: string
  url: string
  inline: boolean
}

export interface PendingToolCall {
  id: string
  name: string
  args: Record<string, unknown>
}

/** One SSE event from POST /api/assistant/chat. */
export type AssistantEvent =
  | { type: 'start'; conversation_id: string; provider: string; model: string }
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  | {
      type: 'tool_call'
      id: string
      name: string
      args: Record<string, unknown>
      side: 'server' | 'client'
    }
  | { type: 'tool_result'; id: string; name: string; ok: boolean; summary: string }
  | { type: 'artifact'; [k: string]: unknown }
  | { type: 'await_client'; calls: PendingToolCall[] }
  | { type: 'usage'; input_tokens: number; output_tokens: number }
  | { type: 'notice'; message: string }
  | { type: 'error'; message: string }
  | { type: 'done'; reason: string }

/** What the transcript renders, in order. */
export type TranscriptItem =
  | { kind: 'user'; text: string; files: string[] }
  | { kind: 'text'; text: string }
  | { kind: 'thinking'; text: string }
  | {
      kind: 'tool'
      id: string
      name: string
      args: Record<string, unknown>
      side: 'server' | 'client'
      status: 'running' | 'ok' | 'error'
      summary: string
    }
  | { kind: 'artifact'; artifact: AssistantArtifact }
  | { kind: 'notice'; message: string }
  | { kind: 'error'; message: string }

export interface ClientToolOutcome {
  id: string
  ok: boolean
  result?: unknown
  error?: string
}
