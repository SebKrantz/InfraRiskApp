// The chat panel: overlays the right side of the map above the map controls
// (z-10) and the sidebar dropdown (z-30), below the modal dialogs (z-50).

import { useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { Eraser, Loader2, Paperclip, Send, Square, X } from 'lucide-react'

import type { UseAssistant } from '../../hooks/useAssistant'
import type { AssistantMeta } from '../../types/assistant'
import MessageList from './MessageList'

export default function AssistantPanel({
  assistant,
  meta,
  onClose,
}: {
  assistant: UseAssistant
  meta: AssistantMeta
  onClose: () => void
}) {
  const { items, streaming, send, stop, clear } = assistant
  const [text, setText] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  const providers = meta.providers.filter((p) => p.available)
  const [choice, setChoice] = useState(() => {
    const p = providers.find((x) => x.id === meta.default_provider) ?? providers[0]
    return p ? `${p.id}:${p.default_model}` : ''
  })
  const fileInput = useRef<HTMLInputElement>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    if (streaming || (!text.trim() && files.length === 0)) return
    const [provider, model] = choice.split(':')
    send(text.trim(), files, provider, model)
    setText('')
    setFiles([])
    if (textarea.current) textarea.current.style.height = 'auto'
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const addFiles = (list: FileList | null) => {
    if (!list) return
    setFiles((prev) => [...prev, ...Array.from(list)].slice(0, 8))
  }

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    addFiles(e.target.files)
    e.target.value = ''
  }

  return (
    <div
      className="absolute inset-y-0 right-0 z-30 flex w-[400px] max-w-[85vw] flex-col border-l border-gray-700 bg-gray-900/95 backdrop-blur-[3px]"
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        addFiles(e.dataTransfer.files)
      }}
    >
      {/* header */}
      <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-2">
        <span className="text-sm font-semibold text-gray-100">Assistant</span>
        <select
          value={choice}
          onChange={(e) => setChoice(e.target.value)}
          disabled={streaming}
          className="min-w-0 flex-1 rounded-md border border-gray-700 bg-gray-800 px-2 py-1 text-[11px] text-gray-300 disabled:opacity-50"
        >
          {providers.map((p) =>
            p.models.map((m) => (
              <option key={`${p.id}:${m}`} value={`${p.id}:${m}`}>
                {p.label} · {m}
              </option>
            )),
          )}
        </select>
        <button
          type="button"
          title="Clear conversation"
          onClick={clear}
          disabled={streaming || items.length === 0}
          className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200 disabled:opacity-40"
        >
          <Eraser className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Close"
          onClick={onClose}
          className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <MessageList items={items} streaming={streaming} />

      {/* composer */}
      <div
        className={`border-t border-gray-800 p-2 ${
          dragOver ? 'rounded-md outline-dashed outline-2 outline-blue-500/60' : ''
        }`}
      >
        {files.length > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1">
            {files.map((f, i) => (
              <span
                key={`${f.name}-${i}`}
                className="inline-flex items-center gap-1 rounded-full border border-gray-700 bg-gray-800 px-2 py-0.5 text-[10px] text-gray-300"
              >
                <Paperclip className="h-2.5 w-2.5" />
                {f.name}
                <button
                  type="button"
                  onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                  className="text-gray-500 hover:text-gray-300"
                >
                  <X className="h-2.5 w-2.5" />
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="flex items-end gap-1.5">
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".gpkg,.zip,.geojson,.shp,.csv,.xlsx,.xls,.md,.markdown,.txt,.docx,.json,.png,.jpg,.jpeg,.webp,.pdf"
            className="hidden"
            onChange={onPick}
          />
          <button
            type="button"
            title="Attach infrastructure data, vulnerability curves or context documents"
            onClick={() => fileInput.current?.click()}
            className="shrink-0 rounded p-1.5 text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <textarea
            ref={textarea}
            value={text}
            rows={1}
            placeholder="Ask about exposure, damage or hazards…"
            onChange={(e) => {
              setText(e.target.value)
              e.currentTarget.style.height = 'auto'
              e.currentTarget.style.height = `${Math.min(e.currentTarget.scrollHeight, 140)}px`
            }}
            onKeyDown={onKeyDown}
            className="min-h-[34px] flex-1 resize-none rounded-md border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
          />
          {streaming ? (
            <button
              type="button"
              title="Stop"
              onClick={stop}
              className="shrink-0 rounded-md border border-gray-600 p-1.5 text-gray-300 hover:bg-gray-800"
            >
              <Square className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              type="button"
              title="Send"
              onClick={submit}
              disabled={!text.trim() && files.length === 0}
              className="shrink-0 rounded-md bg-blue-600 p-1.5 text-white hover:bg-blue-500 disabled:opacity-40"
            >
              <Send className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        {streaming && (
          <div className="mt-1 flex items-center gap-1 text-[10px] text-gray-500">
            <Loader2 className="h-3 w-3 animate-spin" /> working — hazard rasters can
            take a minute
          </div>
        )}
      </div>
    </div>
  )
}
