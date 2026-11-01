// The sidebar's no-data line: how much of an analysis had no data, and what that means for
// the layer. No data counts as unaffected, so it sits inside the "unaffected" bar; the line
// says how much of the result that is (the server's nodata_* summary fields).

interface NodataSummary {
  nodata_samples?: number
  nodata_share_pct?: number
  nodata_km?: number
  nodata_note?: string
}

const fmt = (value: number, digits: number) =>
  value.toLocaleString('en-US', { maximumFractionDigits: digits })

export function nodataLine(summary: NodataSummary, geometryType: string): string | null {
  if (!summary.nodata_samples) return null
  const of = geometryType === 'LineString' ? 'sample points' : 'assets'
  const km = summary.nodata_km !== undefined ? ` (${fmt(summary.nodata_km, 1)} km)` : ''
  const note = summary.nodata_note
    ? ` ${summary.nodata_note.charAt(0).toUpperCase()}${summary.nodata_note.slice(1)}.`
    : ''
  return `No data at ${fmt(summary.nodata_share_pct ?? 0, 2)}% of ${of}${km}.${note}`
}
