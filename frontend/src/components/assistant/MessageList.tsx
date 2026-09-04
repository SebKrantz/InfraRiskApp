// The transcript: user bubbles, streamed assistant markdown, tool chips,
// artifacts, errors. Sticks to the bottom while streaming unless the user has
// scrolled up.

import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, Paperclip, Sparkles } from 'lucide-react'

import type { TranscriptItem } from '../../types/assistant'
import ArtifactCard from './ArtifactCard'
import Markdown from './Markdown'
import ToolChip from './ToolChip'

function Thinking({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="my-1">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-gray-500 hover:text-gray-400"
      >
        <Sparkles className="h-3 w-3" />
        thinking
        {open ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>
      {open && (
        <div className="mt-1 max-h-40 overflow-y-auto rounded-md border border-gray-800 bg-gray-900 p-2 text-[11px] italic leading-relaxed text-gray-500">
          {text}
        </div>
      )}
    </div>
  )
}

export default function MessageList({
  items,
  streaming,
}: {
  items: TranscriptItem[]
  streaming: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [items])

  return (
    <div
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60
      }}
      className="flex-1 space-y-1 overflow-y-auto px-3 py-3"
    >
      {items.length === 0 && (
        <div className="mt-8 space-y-2 px-2 text-center text-xs text-gray-500">
          <Sparkles className="mx-auto h-6 w-6 text-gray-600" />
          <p className="font-medium text-gray-400">
            Attach infrastructure data and ask
          </p>
          <p>
            "How much of this network is in the 100-year flood zone?" · "Compare
            exposure across return periods" · "Run a damage analysis with this
            curve at $40/m and write it up as a Word report"
          </p>
        </div>
      )}
      {items.map((item, i) => {
        switch (item.kind) {
          case 'user':
            return (
              <div key={i} className="flex justify-end pb-1 pt-2">
                <div className="max-w-[85%] rounded-lg rounded-br-sm bg-blue-600 px-3 py-1.5 text-xs text-white">
                  {item.text}
                  {item.files.length > 0 && (
                    <span className="mt-1 flex flex-wrap gap-1">
                      {item.files.map((f) => (
                        <span
                          key={f}
                          className="inline-flex items-center gap-1 rounded bg-blue-700/70 px-1.5 py-0.5 text-[10px]"
                        >
                          <Paperclip className="h-2.5 w-2.5" />
                          {f}
                        </span>
                      ))}
                    </span>
                  )}
                </div>
              </div>
            )
          case 'text':
            return <Markdown key={i} text={item.text} />
          case 'thinking':
            return <Thinking key={i} text={item.text} />
          case 'tool':
            return <ToolChip key={item.id} item={item} />
          case 'artifact':
            return <ArtifactCard key={item.artifact.id} artifact={item.artifact} />
          case 'notice':
            return (
              <div key={i} className="my-1 text-[11px] italic text-amber-400/80">
                {item.message}
              </div>
            )
          case 'error':
            return (
              <div
                key={i}
                className="my-1 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300"
              >
                {item.message}
              </div>
            )
        }
      })}
      {streaming && items[items.length - 1]?.kind !== 'tool' && (
        <div className="flex items-center gap-1 py-1 text-gray-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-500" />
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-500 [animation-delay:150ms]" />
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-500 [animation-delay:300ms]" />
        </div>
      )}
    </div>
  )
}
