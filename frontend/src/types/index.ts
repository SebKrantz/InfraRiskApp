export interface Hazard {
  id: string
  name: string
  url: string
  description?: string
  metadata?: string
  unit?: string
  /** Topical group for UI (e.g. optgroup in hazard dropdown); from CSV column `category` */
  category?: string
}

export interface UploadedFile {
  file_id: string
  filename: string
  geometry_type: 'Point' | 'LineString'
  feature_count: number
  crs: string | null
  bounds: {
    minx: number
    miny: number
    maxx: number
    maxy: number
  }
  geojson?: any  // Optional GeoJSON for initial display
}

export interface AnalysisResult {
  file_id: string
  hazard_id: string
  geometry_type: 'Point' | 'LineString'
  summary: {
    total_features: number
    affected_count?: number
    unaffected_count?: number
    affected_meters?: number
    unaffected_meters?: number
    /** Sample points (features for points, 100 m points along a line) and how many had no data;
     *  no data counts as unaffected, so it is part of the unaffected figures */
    samples?: number
    nodata_samples?: number
    nodata_share_pct?: number
    /** Lines: length of the spans with no data at either end, and its share of the length */
    nodata_km?: number
    nodata_length_share_pct?: number | null
    /** What no data means for this layer and threshold (present when there is any) */
    nodata_note?: string
    total_damage_cost?: number
    /** Lower-bound damage cost (from 4-column vulnerability CSV with uncertainty bands) */
    total_damage_cost_lower?: number
    /** Upper-bound damage cost (from 4-column vulnerability CSV with uncertainty bands) */
    total_damage_cost_upper?: number
  }
  affected_features?: any
  infrastructure_features?: any // Full GeoJSON with affected status (includes vulnerability and damage_cost properties)
}

export type ColorPalette = 'viridis' | 'magma' | 'inferno' | 'plasma' | 'cividis' | 'turbo'

export type Basemap = 'positron' | 'dark-matter' | 'osm' | 'topo' | 'esri-street' | 'esri-topo' | 'esri-terrain' | 'esri-ocean' | 'esri-imagery'

