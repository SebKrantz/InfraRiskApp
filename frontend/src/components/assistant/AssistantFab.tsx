// The toggle: a floating button in the bottom-right of the map area.

import { Sparkles } from 'lucide-react'

export default function AssistantFab({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      title="AI assistant"
      onClick={onClick}
      className="absolute bottom-8 right-4 z-20 flex h-11 w-11 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg shadow-blue-900/40 transition-colors hover:bg-blue-500"
    >
      <Sparkles className="h-5 w-5" />
    </button>
  )
}
