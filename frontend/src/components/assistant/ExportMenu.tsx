// The panel's Export menu: the whole conversation, for debugging — every message with its
// tool calls and full results, thinking, errors, time, model and tokens. Copied as Markdown or
// downloaded as Markdown / JSON; the server builds it (backend/app/assistant/transcript.py).

import { useEffect, useRef, useState } from 'react'
import { Braces, Check, Copy, FileText, Loader2, Share } from 'lucide-react'

import { exportMarkdown, exportUrl } from '../../lib/assistantApi'

const itemCls =
  'flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-[12px] text-gray-200 transition-colors hover:bg-gray-800 disabled:text-gray-500'

export default function ExportMenu({
  conversationId,
  disabled,
}: {
  conversationId: string | null
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  // Fetched when the menu opens, so the copy itself runs inside the click (Safari refuses a
  // clipboard write that waits on a network request first).
  const [markdown, setMarkdown] = useState<string | null>(null)
  const [state, setState] = useState<'idle' | 'copied' | string>('idle')
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

  const toggle = () => {
    if (open || !conversationId) return setOpen(false)
    setOpen(true)
    setMarkdown(null)
    setState('idle')
    exportMarkdown(conversationId).then(setMarkdown, (e) =>
      setState(e instanceof Error ? e.message : String(e)),
    )
  }

  const copy = () => {
    if (markdown === null) return
    if (!navigator.clipboard) return setState('Copying is not available here; use a download.')
    navigator.clipboard.writeText(markdown).then(
      () => {
        setState('copied')
        setTimeout(() => setOpen(false), 700)
      },
      (e) => setState(e instanceof Error ? e.message : String(e)),
    )
  }

  const failed = state !== 'idle' && state !== 'copied'
  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        title="Copy or export this conversation"
        aria-label="Export conversation"
        onClick={toggle}
        disabled={disabled || !conversationId}
        className={`rounded p-1 hover:bg-gray-800 hover:text-gray-200 disabled:opacity-40 ${
          open ? 'text-blue-400' : 'text-gray-400'
        }`}
      >
        <Share className="h-3.5 w-3.5" />
      </button>
      {open && conversationId && (
        <div className="absolute right-0 top-full z-40 mt-1 w-60 overflow-hidden rounded-md border border-gray-700 bg-gray-900 py-1 shadow-xl shadow-black/50">
          <div className="px-3 pb-1 pt-1 text-[10px] font-medium uppercase tracking-widest text-gray-500">
            Whole conversation
          </div>
          <button type="button" onClick={copy} disabled={markdown === null} className={itemCls}>
            {state === 'copied' ? (
              <Check className="h-3.5 w-3.5 shrink-0 text-green-400" />
            ) : markdown === null && !failed ? (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-gray-500" />
            ) : (
              <Copy className="h-3.5 w-3.5 shrink-0 text-gray-400" />
            )}
            {state === 'copied' ? 'Copied' : 'Copy as Markdown'}
          </button>
          <a
            href={exportUrl(conversationId, 'md')}
            download
            onClick={() => setOpen(false)}
            className={itemCls}
          >
            <FileText className="h-3.5 w-3.5 shrink-0 text-gray-400" />
            Download Markdown
            <span className="ml-auto font-mono text-[10.5px] text-gray-500">.md</span>
          </a>
          <a
            href={exportUrl(conversationId, 'json')}
            download
            onClick={() => setOpen(false)}
            className={itemCls}
          >
            <Braces className="h-3.5 w-3.5 shrink-0 text-gray-400" />
            Download JSON
            <span className="ml-auto font-mono text-[10.5px] text-gray-500">.json</span>
          </a>
          {failed && (
            <div className="px-3 pb-1.5 pt-1 text-[11px] leading-snug text-red-400">{state}</div>
          )}
        </div>
      )}
    </div>
  )
}
