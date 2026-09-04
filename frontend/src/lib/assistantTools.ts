// The browser side of the assistant's tool protocol: the ui_* executor.
//
// The loop suspends with `await_client`; useAssistant runs each call through
// execute() against the bindings App.tsx refreshes every render (a ref, so the
// assistant's stable actions object never goes stale).

import type { MutableRefObject } from 'react'

import type {
  AnalysisResult,
  Basemap,
  ColorPalette,
  Hazard,
  UploadedFile,
} from '../types'
import type { ClientToolOutcome, PendingToolCall } from '../types/assistant'

/** Everything the executor may read or call, refreshed by App.tsx each render. */
export interface AssistantBindings {
  // state (read by the snapshot)
  sidebarOpen: boolean
  uploadedFile: UploadedFile | null
  selectedHazard: Hazard | null
  analysisResult: AnalysisResult | null
  colorPalette: ColorPalette
  intensityThreshold: number
  hazardOpacity: number
  hazardStats: { min: number; max: number } | null
  statsHazardId: string | null
  basemap: Basemap
  hazards: Hazard[]
  loadingAnalysis: boolean
  error: string | null
  vulnerabilityAnalysisEnabled: boolean
  vulnerabilityCurveFile: File | null
  replacementValue: number | null
  // actions
  setSidebarOpen: (v: boolean) => void
  setUploadedFile: (f: UploadedFile | null) => void
  setSelectedHazard: (h: Hazard | null) => void
  setColorPalette: (p: ColorPalette) => void
  setIntensityThreshold: (t: number) => void
  setHazardOpacity: (o: number) => void
  setBasemap: (b: Basemap) => void
  setVulnerabilityAnalysisEnabled: (v: boolean) => void
  setVulnerabilityCurveFile: (f: File | null) => void
  setReplacementValue: (v: number | null) => void
  clearData: () => void
  fitBounds: (bbox: [number, number, number, number]) => boolean
}

export type BindingsRef = MutableRefObject<AssistantBindings>

const PALETTES: ColorPalette[] = [
  'viridis', 'magma', 'inferno', 'plasma', 'cividis', 'turbo',
]
const BASEMAPS: Basemap[] = [
  'positron', 'dark-matter', 'osm', 'topo', 'esri-street', 'esri-topo',
  'esri-terrain', 'esri-ocean', 'esri-imagery', 'google-maps',
  'google-terrain', 'google-hybrid', 'google-satellite',
]

const asStr = (v: unknown): string | undefined =>
  typeof v === 'string' && v ? v : undefined
const asNum = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined
const asBool = (v: unknown): boolean | undefined =>
  typeof v === 'boolean' ? v : undefined

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** Wait until `check` passes, polling the live bindings. */
async function waitFor(
  ref: BindingsRef,
  check: (b: AssistantBindings) => boolean,
  timeoutMs = 20000,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (check(ref.current)) return true
    await sleep(100)
  }
  return false
}

/** Resolve a hazard by id, exact name, or unique substring. */
function resolveHazard(hazards: Hazard[], ref: string): Hazard {
  const needle = ref.trim().toLowerCase()
  const byId = hazards.find((h) => h.id.toLowerCase() === needle)
  if (byId) return byId
  const byName = hazards.find((h) => h.name.toLowerCase() === needle)
  if (byName) return byName
  const subs = hazards.filter((h) => h.name.toLowerCase().includes(needle))
  if (subs.length === 1) return subs[0]
  if (subs.length > 1)
    throw new Error(
      `"${ref}" matches ${subs.length} layers: ${subs.slice(0, 4).map((h) => h.name).join(', ')}`,
    )
  throw new Error(`no hazard layer matching "${ref}"`)
}

/** The ui_read_app_state payload — and the debounced push MCP clients read. */
export function snapshot(b: AssistantBindings): Record<string, unknown> {
  return {
    sidebar_open: b.sidebarOpen,
    dataset: b.uploadedFile
      ? {
          file_id: b.uploadedFile.file_id,
          filename: b.uploadedFile.filename,
          geometry_type: b.uploadedFile.geometry_type,
          feature_count: b.uploadedFile.feature_count,
          bounds: b.uploadedFile.bounds,
        }
      : null,
    hazard: b.selectedHazard
      ? {
          hazard_id: b.selectedHazard.id,
          name: b.selectedHazard.name,
          unit: b.selectedHazard.unit,
          category: b.selectedHazard.category,
        }
      : null,
    threshold: b.intensityThreshold,
    threshold_range: b.hazardStats,
    threshold_is_at_minimum: b.hazardStats
      ? b.intensityThreshold <= b.hazardStats.min
      : null,
    vulnerability: {
      enabled: b.vulnerabilityAnalysisEnabled,
      curve_file: b.vulnerabilityCurveFile?.name ?? null,
      replacement_value: b.replacementValue,
    },
    display: {
      palette: b.colorPalette,
      hazard_opacity: b.hazardOpacity,
      basemap: b.basemap,
    },
    result: b.analysisResult
      ? { geometry_type: b.analysisResult.geometry_type, ...b.analysisResult.summary }
      : null,
    running: b.loadingAnalysis,
    error: b.error,
    hazard_layers_available: b.hazards.length,
  }
}

async function run(
  ref: BindingsRef,
  name: string,
  args: Record<string, unknown>,
  conversationId: string | null,
): Promise<unknown> {
  const b = ref.current
  switch (name) {
    case 'ui_read_app_state':
      return snapshot(b)

    case 'ui_show_dataset': {
      const fileId = asStr(args.file_id)
      if (!fileId) throw new Error('file_id is required')
      const res = await fetch(`/api/assistant/datasets/${encodeURIComponent(fileId)}`)
      if (!res.ok) throw new Error(`dataset ${fileId} not found on the server`)
      const data = (await res.json()) as UploadedFile
      b.setUploadedFile(data)
      return `showing ${data.filename} (${data.feature_count} ${data.geometry_type} features); the map fits to it and the analysis re-runs if a hazard is selected`
    }

    case 'ui_select_hazard': {
      const hazardRef = asStr(args.hazard)
      if (hazardRef === undefined) {
        b.setSelectedHazard(null)
        return 'cleared the hazard selection'
      }
      const hazard = resolveHazard(b.hazards, hazardRef)
      b.setSelectedHazard(hazard)
      // Selecting a layer kicks off a stats fetch that overwrites the
      // threshold; wait for it to land so a following ui_set_threshold sticks.
      const settled = await waitFor(ref, (cur) => cur.statsHazardId === hazard.id)
      const stats = ref.current.hazardStats
      if (!settled || !stats)
        return `selected ${hazard.name}; its intensity range is still loading`
      return `selected ${hazard.name}; intensity ranges ${stats.min} to ${stats.max} ${hazard.unit ?? ''}, threshold reset to ${stats.min}`
    }

    case 'ui_set_threshold': {
      const threshold = asNum(args.threshold)
      if (threshold === undefined) throw new Error('threshold must be a number')
      if (!b.selectedHazard) throw new Error('select a hazard first')
      // If a hazard was just picked, its stats fetch may still be in flight and
      // would clobber this value on arrival.
      if (b.statsHazardId !== b.selectedHazard.id)
        await waitFor(ref, (cur) => cur.statsHazardId === cur.selectedHazard?.id)
      const stats = ref.current.hazardStats
      if (stats && (threshold < stats.min || threshold > stats.max))
        throw new Error(
          `threshold ${threshold} is outside this layer's range (${stats.min} to ${stats.max})`,
        )
      ref.current.setIntensityThreshold(threshold)
      return `threshold set to ${threshold}${b.selectedHazard.unit ? ` ${b.selectedHazard.unit}` : ''}; the analysis re-runs`
    }

    case 'ui_set_vulnerability': {
      const enabled = asBool(args.enabled)
      if (enabled === undefined) throw new Error('enabled must be true or false')
      if (!enabled) {
        b.setVulnerabilityAnalysisEnabled(false)
        return 'vulnerability mode off; back to plain exposure'
      }
      const curve = asStr(args.curve)
      const value = asNum(args.replacement_value)
      if (curve) {
        if (!conversationId) throw new Error('no conversation to fetch the curve from')
        const res = await fetch(
          `/api/assistant/conversations/${encodeURIComponent(conversationId)}/curves/${encodeURIComponent(curve)}`,
        )
        if (!res.ok) throw new Error(`curve "${curve}" is not loaded on the server`)
        const blob = await res.blob()
        b.setVulnerabilityCurveFile(new File([blob], `${curve}.csv`, { type: 'text/csv' }))
      }
      if (value !== undefined) {
        if (value <= 0) throw new Error('replacement_value must be greater than zero')
        b.setReplacementValue(value)
      }
      b.setVulnerabilityAnalysisEnabled(true)
      const haveCurve = curve || b.vulnerabilityCurveFile
      const haveValue = value !== undefined || b.replacementValue
      if (!haveCurve || !haveValue)
        return 'vulnerability mode on, but the app needs BOTH a curve and a positive replacement value before it will compute anything'
      return `vulnerability mode on with curve ${curve ?? b.vulnerabilityCurveFile?.name} and replacement value ${value ?? b.replacementValue}; the analysis re-runs`
    }

    case 'ui_set_display': {
      const done: string[] = []
      const palette = asStr(args.palette)
      if (palette) {
        if (!PALETTES.includes(palette as ColorPalette))
          throw new Error(`unknown palette; one of: ${PALETTES.join(', ')}`)
        b.setColorPalette(palette as ColorPalette)
        done.push(`palette=${palette}`)
      }
      const opacity = asNum(args.opacity)
      if (opacity !== undefined) {
        if (opacity < 0 || opacity > 100) throw new Error('opacity must be 0-100')
        b.setHazardOpacity(opacity)
        done.push(`opacity=${opacity}%`)
      }
      const basemap = asStr(args.basemap)
      if (basemap) {
        if (!BASEMAPS.includes(basemap as Basemap))
          throw new Error(`unknown basemap; one of: ${BASEMAPS.join(', ')}`)
        b.setBasemap(basemap as Basemap)
        done.push(`basemap=${basemap}`)
      }
      return done.length ? done.join(', ') : 'nothing changed'
    }

    case 'ui_fit_map': {
      const bbox = args.bbox
      if (
        !Array.isArray(bbox) ||
        bbox.length !== 4 ||
        !bbox.every((v) => typeof v === 'number' && Number.isFinite(v))
      )
        throw new Error('bbox must be [west, south, east, north] numbers')
      const ok = b.fitBounds(bbox as [number, number, number, number])
      if (!ok) throw new Error('the map is not ready yet')
      return `moved the map to [${(bbox as number[]).map((v) => v.toFixed(2)).join(', ')}]`
    }

    case 'ui_set_sidebar': {
      const open = asBool(args.open)
      if (open === undefined) throw new Error('open must be true or false')
      b.setSidebarOpen(open)
      return open ? 'sidebar opened' : 'sidebar collapsed'
    }

    case 'ui_clear_data': {
      b.clearData()
      return 'cleared the dataset and its results'
    }

    default:
      throw new Error(`unknown client tool ${name}`)
  }
}

export async function executeClientTool(
  ref: BindingsRef,
  call: PendingToolCall,
  conversationId: string | null,
): Promise<ClientToolOutcome> {
  try {
    const result = await run(ref, call.name, call.args ?? {}, conversationId)
    return { id: call.id, ok: true, result }
  } catch (e) {
    return { id: call.id, ok: false, error: (e as Error).message }
  }
}
