// One tool call in the transcript: a status pill, expandable to args + result.

import { useState } from 'react'
import { Check, ChevronDown, ChevronUp, Loader2, Monitor, Server, X } from 'lucide-react'

import type { TranscriptItem } from '../../types/assistant'

type ToolItem = Extract<TranscriptItem, { kind: 'tool' }>

export default function ToolChip({ item }: { item: ToolItem }) {
  const [open, setOpen] = useState(false)
  const hasDetail = Object.keys(item.args ?? {}).length > 0 || item.summary !== ''
  const base =
    item.status === 'error'
      ? 'border-red-500/40 bg-red-500/10 text-red-300'
      : 'border-gray-700 bg-gray-800/80 text-gray-300'
  return (
    <div className="my-1">
      <button
        type="button"
        onClick={() => hasDetail && setOpen(!open)}
        className={`flex max-w-full items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${base} ${
          hasDetail ? 'cursor-pointer hover:border-gray-600' : ''
        }`}
      >
        {item.status === 'running' ? (
          <Loader2 className="h-3 w-3 shrink-0 animate-spin text-blue-400" />
        ) : item.status === 'ok' ? (
          <Check className="h-3 w-3 shrink-0 text-emerald-400" />
        ) : (
          <X className="h-3 w-3 shrink-0 text-red-400" />
        )}
        {item.side === 'client' ? (
          <Monitor className="h-3 w-3 shrink-0 text-gray-500" />
        ) : (
          <Server className="h-3 w-3 shrink-0 text-gray-500" />
        )}
        <span className="truncate font-mono">{item.name}</span>
        {hasDetail &&
          (open ? (
            <ChevronUp className="h-3 w-3 shrink-0 text-gray-500" />
          ) : (
            <ChevronDown className="h-3 w-3 shrink-0 text-gray-500" />
          ))}
      </button>
      {open && (
        <div className="mt-1 space-y-1 rounded-md border border-gray-700/70 bg-gray-800/60 p-2 text-[11px] text-gray-400">
          {Object.keys(item.args ?? {}).length > 0 && (
            <pre className="overflow-x-auto font-mono leading-snug">
              {JSON.stringify(item.args, null, 1)}
            </pre>
          )}
          {item.summary && <div className="text-gray-300">{item.summary}</div>}
        </div>
      )}
    </div>
  )
}
