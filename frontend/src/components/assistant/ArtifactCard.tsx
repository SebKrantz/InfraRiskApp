// A generated file in the transcript: inline image for charts and maps, a
// download card for documents and data.

import {
  BarChart3,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Map as MapIcon,
  Table,
} from 'lucide-react'

import type { AssistantArtifact } from '../../types/assistant'

const ICONS = {
  chart: BarChart3,
  map: MapIcon,
  png: BarChart3,
  docx: FileText,
  xlsx: FileSpreadsheet,
  csv: Table,
  gpkg: Database,
  file: FileText,
} as const

export default function ArtifactCard({ artifact }: { artifact: AssistantArtifact }) {
  if (artifact.inline) {
    return (
      <a
        href={artifact.url}
        target="_blank"
        rel="noreferrer"
        title={artifact.title}
        className="my-1.5 block overflow-hidden rounded-md border border-gray-700 bg-white"
      >
        <img src={artifact.url} alt={artifact.title} className="max-h-72 w-full object-contain" />
      </a>
    )
  }
  const Icon = ICONS[artifact.kind] ?? FileText
  return (
    <a
      href={artifact.url}
      download={artifact.filename}
      className="my-1.5 flex items-center gap-2.5 rounded-md border border-gray-700 bg-gray-800/80 px-3 py-2 text-xs text-gray-200 hover:border-gray-600 hover:bg-gray-800"
    >
      <Icon className="h-5 w-5 shrink-0 text-blue-400" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">{artifact.filename}</span>
        <span className="block truncate text-[10px] uppercase tracking-wide text-gray-500">
          {artifact.kind}
        </span>
      </span>
      <Download className="h-4 w-4 shrink-0 text-gray-500" />
    </a>
  )
}
