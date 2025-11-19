export interface Hazard {
  id: string
  name: string
  url: string
  description?: string
  metadata?: string
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
  }
  affected_features?: any
  infrastructure_features?: any // Full GeoJSON with affected status
}

export type ColorPalette = 'viridis' | 'magma' | 'inferno' | 'plasma' | 'cividis' | 'turbo'

export type Basemap = 'positron' | 'dark-matter' | 'osm' | 'topo' | 'esri-street' | 'esri-topo' | 'esri-terrain' | 'esri-ocean' | 'esri-imagery' | 'google-maps' | 'google-terrain' | 'google-hybrid' | 'google-satellite'

