// Types for the AI assistant: availability, SSE events, transcript, client tools.

// Served by GET /api/assistant/meta from backend/app/assistant/models.py (the one table;
// keep it in step with AGUI backend/agui/config.py).
export interface AssistantProvider {
  id: string
  label: string
  models: string[]
  default_model: string
  available: boolean
  /** reasoning levels per model; an empty list: the model takes none */
  efforts: Record<string, string[]>
  /** the level shown as "Default (x)" per model; null: no effort control */
  effort_default: Record<string, string | null>
  /** "standard", and "flex" where the provider has it */
  tiers: string[]
}

/** The assistant's current setting: what the next chat request carries. */
export interface AssistantChoice {
  provider: string
  model: string
  /** null: the model's default */
  effort: string | null
  tier: 'standard' | 'flex'
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
