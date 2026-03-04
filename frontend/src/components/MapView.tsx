import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { Hazard, UploadedFile, AnalysisResult, ColorPalette, Basemap } from '../types'
import { Select } from './ui/select'

interface MapViewProps {
  uploadedFile: UploadedFile | null
  selectedHazard: Hazard | null
  analysisResult: AnalysisResult | null
  colorPalette: ColorPalette
  intensityThreshold: number
  hazardOpacity: number
  basemap: Basemap
  onBasemapChange: (basemap: Basemap) => void
  loadingAnalysis?: boolean
  vulnerabilityAnalysisEnabled?: boolean
  hazardStats?: { min: number; max: number } | null
}

const basemapStyles: Record<Basemap, any> = {
  positron: {
    version: 8,
    sources: {
      'carto-positron': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
          'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© OpenStreetMap © CARTO'
      }
    },
    layers: [
      {
        id: 'carto-positron-layer',
        type: 'raster',
        source: 'carto-positron',
        minzoom: 0,
        maxzoom: 22
      }
    ]
  },
  'dark-matter': {
    version: 8,
    sources: {
      'carto-dark-matter': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© OpenStreetMap © CARTO'
      }
    },
    layers: [
      {
        id: 'carto-dark-matter-layer',
        type: 'raster',
        source: 'carto-dark-matter',
        minzoom: 0,
        maxzoom: 22
      }
    ]
  },
  osm: {
    version: 8,
    sources: {
      'osm': {
        type: 'raster',
        tiles: [
          'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© OpenStreetMap contributors'
      }
    },
    layers: [
      {
        id: 'osm-layer',
        type: 'raster',
        source: 'osm',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  },
  topo: {
    version: 8,
    sources: {
      'opentopomap': {
        type: 'raster',
        tiles: [
          'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
          'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
          'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '© OpenTopoMap contributors'
      }
    },
    layers: [
      {
        id: 'opentopomap-layer',
        type: 'raster',
        source: 'opentopomap',
        minzoom: 0,
        maxzoom: 17
      }
    ]
  },
  'esri-street': {
    version: 8,
    sources: {
      'esri-street': {
        type: 'raster',
        tiles: [
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        attribution: '© Esri'
      }
    },
    layers: [
      {
        id: 'esri-street-layer',
        type: 'raster',
        source: 'esri-street',
        minzoom: 0,
        maxzoom: 16
      }
    ]
  },
  'esri-topo': {
    version: 8,
    sources: {
      'esri-topo': {
        type: 'raster',
        tiles: [
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        attribution: '© Esri'
      }
    },
    layers: [
      {
        id: 'esri-topo-layer',
        type: 'raster',
        source: 'esri-topo',
        minzoom: 0,
        maxzoom: 16
      }
    ]
  },
  'esri-terrain': {
    version: 8,
    sources: {
      'esri-terrain': {
        type: 'raster',
        tiles: [
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        attribution: '© Esri'
      }
    },
    layers: [
      {
        id: 'esri-terrain-layer',
        type: 'raster',
        source: 'esri-terrain',
        minzoom: 0,
        maxzoom: 13
      }
    ]
  },
  'esri-ocean': {
    version: 8,
    sources: {
      'esri-ocean': {
        type: 'raster',
        tiles: [
          'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        attribution: '© Esri'
      }
    },
    layers: [
      {
        id: 'esri-ocean-layer',
        type: 'raster',
        source: 'esri-ocean',
        minzoom: 0,
        maxzoom: 16
      }
    ]
  },
  'esri-imagery': {
    version: 8,
    sources: {
      'esri-imagery': {
        type: 'raster',
        tiles: [
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        ],
        tileSize: 256,
        attribution: '© Esri'
      }
    },
    layers: [
      {
        id: 'esri-imagery-layer',
        type: 'raster',
        source: 'esri-imagery',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  },
  'google-maps': {
    version: 8,
    sources: {
      'google-maps': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
        ],
        tileSize: 256,
        attribution: '© Google'
      }
    },
    layers: [
      {
        id: 'google-maps-layer',
        type: 'raster',
        source: 'google-maps',
        minzoom: 0,
        maxzoom: 20
      }
    ]
  },
  'google-terrain': {
    version: 8,
    sources: {
      'google-terrain': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
        ],
        tileSize: 256,
        attribution: '© Google'
      }
    },
    layers: [
      {
        id: 'google-terrain-layer',
        type: 'raster',
        source: 'google-terrain',
        minzoom: 0,
        maxzoom: 20
      }
    ]
  },
  'google-hybrid': {
    version: 8,
    sources: {
      'google-hybrid': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
        ],
        tileSize: 256,
        attribution: '© Google'
      }
    },
    layers: [
      {
        id: 'google-hybrid-layer',
        type: 'raster',
        source: 'google-hybrid',
        minzoom: 0,
        maxzoom: 20
      }
    ]
  },
  'google-satellite': {
    version: 8,
    sources: {
      'google-satellite': {
        type: 'raster',
        tiles: [
          'https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
          'https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        ],
        tileSize: 256,
        attribution: '© Google'
      }
    },
    layers: [
      {
        id: 'google-satellite-layer',
        type: 'raster',
        source: 'google-satellite',
        minzoom: 0,
        maxzoom: 20
      }
    ]
  }
}

const COLORMAP_STOPS: Record<ColorPalette, string[]> = {
  viridis: ['#440154', '#472777', '#3e4989', '#30678d', '#25828e', '#1e9d88', '#35b778', '#6dce58', '#b5dd2b', '#fde724'],
  magma: ['#000003', '#170f3c', '#430f75', '#711f81', '#9e2e7e', '#cd3f70', '#f0605d', '#fd9567', '#fec98d', '#fbfcbf'],
  inferno: ['#000003', '#1a0b40', '#4a0b6a', '#781c6d', '#a42c60', '#cf4446', '#ed6825', '#fb9b06', '#f7d13c', '#fcfea4'],
  plasma: ['#0c0786', '#45039e', '#7200a8', '#9b179e', '#bc3685', '#d7576b', '#ec7853', '#fa9f3a', '#fcc926', '#eff821'],
  cividis: ['#00224d', '#11356f', '#3a486b', '#575d6d', '#6f7073', '#898678', '#a59b73', '#c3b368', '#e1cc54', '#fde737'],
  turbo: ['#30123b', '#4560d6', '#36a8f9', '#1ae4b6', '#71fd5f', '#c8ee33', '#f9ba38', '#f56817', '#c92903', '#7a0402'],
}

export default function MapView({
  uploadedFile,
  selectedHazard,
  analysisResult,
  colorPalette,
  intensityThreshold,
  hazardOpacity,
  basemap,
  onBasemapChange,
  loadingAnalysis = false,
  vulnerabilityAnalysisEnabled = false,
  hazardStats = null,
}: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const isRestoringLayers = useRef(false)
  const popup = useRef<maplibregl.Popup | null>(null)
  
  // Layer visibility state (basemap always on)
  const [infrastructureVisible, setInfrastructureVisible] = useState(true)
  const [hazardVisible, setHazardVisible] = useState(true)

  // Observed max vulnerability across all features (0-1 scale)
  const maxVulnerability = useMemo(() => {
    if (!vulnerabilityAnalysisEnabled || !analysisResult?.infrastructure_features?.features) return 1
    let max = 0
    for (const f of analysisResult.infrastructure_features.features) {
      const v = f.properties?.vulnerability ?? 0
      if (v > max) max = v
    }
    return max || 1
  }, [vulnerabilityAnalysisEnabled, analysisResult])
  
  // Store current prop values for restoration
  const currentProps = useRef({ uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette, vulnerabilityAnalysisEnabled, maxVulnerability })
  
  // Track zoom level and bounds to prevent unnecessary reloads
  const lastZoomLevel = useRef<number | null>(null)
  const lastBounds = useRef<string | null>(null)
  const boundsSetForFile = useRef<string | null>(null)
  const debounceTimer = useRef<NodeJS.Timeout | null>(null)
  
  // Update refs when props change
  useEffect(() => {
    currentProps.current = { uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette, vulnerabilityAnalysisEnabled, maxVulnerability }
  }, [uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette, vulnerabilityAnalysisEnabled, maxVulnerability])

  useEffect(() => {
    if (!mapContainer.current || map.current) return

    // Ensure container has dimensions
    if (!mapContainer.current.offsetWidth || !mapContainer.current.offsetHeight) {
      console.warn('Map container has no dimensions')
    }

    // Initialize map with CartoDB Positron as default
    try {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: basemapStyles.positron,
        center: [0, 20],
        zoom: 2,
        attributionControl: false,
      })

      map.current.addControl(
        new maplibregl.AttributionControl({
          compact: true,
          customAttribution:
            'Created by <a href="https://sebastiankrantz.com/" target="_blank" rel="noopener noreferrer">Sebastian Krantz</a>, funded by the World Bank',
        }),
        'bottom-right'
      )

      map.current.on('load', () => {
        console.log('Map loaded successfully')
        setMapLoaded(true)
        // Track initial zoom level
        if (map.current) {
          lastZoomLevel.current = map.current.getZoom()
        }
      })

      map.current.on('error', (e: any) => {
        console.error('Map error:', e.error || e)
      })

      map.current.on('style.load', () => {
        console.log('Map style loaded')
        setMapLoaded(true)
        // Track zoom level after style load
        if (map.current) {
          lastZoomLevel.current = map.current.getZoom()
        }
      })

      map.current.on('style.error', (e: any) => {
        console.error('Map style error:', e)
      })
      
      // Track zoom changes to detect when tiles should reload
      map.current.on('zoom', () => {
        if (map.current) {
          const currentZoom = Math.floor(map.current.getZoom())
          if (lastZoomLevel.current !== null && Math.floor(lastZoomLevel.current) !== currentZoom) {
            // Zoom level changed - tiles will reload naturally, just track it
            lastZoomLevel.current = currentZoom
          }
        }
      })
    } catch (error) {
      console.error('Error initializing map:', error)
    }

    return () => {
      // Clean up popup
      if (popup.current) {
        popup.current.remove()
        popup.current = null
      }
      
      // Clean up debounce timer
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
        debounceTimer.current = null
      }
      
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, [])

  // Format feature properties for popup display
  const formatFeatureProperties = (properties: Record<string, any>): string => {
    if (!properties || Object.keys(properties).length === 0) {
      return '<div class="p-2 text-sm text-gray-500">No properties available</div>'
    }

    const formatValue = (value: any): string => {
      if (value === null || value === undefined) {
        return '<span class="text-gray-400 italic">null</span>'
      }
      if (typeof value === 'number') {
        // Format numbers with appropriate precision
        if (Number.isInteger(value)) {
          return value.toLocaleString('en-US')
        }
        // For decimals, show up to 6 decimal places but remove trailing zeros
        const formatted = value.toFixed(6).replace(/\.?0+$/, '')
        return parseFloat(formatted).toLocaleString('en-US')
      }
      if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No'
      }
      if (typeof value === 'string') {
        // Escape HTML and truncate very long strings
        const escaped = value
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;')
        if (escaped.length > 100) {
          return escaped.substring(0, 100) + '...'
        }
        return escaped
      }
      return String(value)
    }

    const formatExposureLevel = (value: any, unit?: string): string => {
      if (value === null || value === undefined) {
        return '<span class="text-gray-400 italic">N/A</span>'
      }
      if (typeof value === 'number') {
        // Format exposure levels with 2-4 decimal places
        // Show 4 decimal places, but remove trailing zeros
        const formatted = value.toFixed(4).replace(/\.?0+$/, '')
        const base = parseFloat(formatted).toLocaleString('en-US', {
          minimumFractionDigits: 0,
          maximumFractionDigits: 4
        })
        return unit ? `${base} ${unit}` : base
      }
      return String(value)
    }

    const formatPropertyName = (key: string): string => {
      // Convert snake_case and camelCase to Title Case
      return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, (str) => str.toUpperCase())
        .trim()
    }

    let html = '<div style="max-width: 300px; font-family: system-ui, -apple-system, sans-serif;">'
    html += '<div class="text-sm font-semibold text-gray-800 mb-1.5 pb-1.5 border-b border-gray-200">Feature Properties</div>'
    html += '<div class="space-y-0.5">'

    // Extract affected status and exposure level properties to display prominently at the top
    const lengthM = properties['length_m']
    const affected = properties['affected']
    const exposureLevel = properties['exposure_level']
    const exposureLevelMax = properties['exposure_level_max']
    const exposureLevelAvg = properties['exposure_level_avg']
    const vulnerability = properties['vulnerability']
    const damageCost = properties['damage_cost']
    
    // Build list of computed features to display in a single shaded box
    const computedFeatures: Array<{label: string, value: string}> = []
    
    // Display length first (for line segments)
    if (lengthM !== undefined && lengthM !== null) {
      const formatLength = (value: number): string => {
        // Format length in meters with appropriate precision
        if (value >= 1000) {
          // Show in km with 2 decimal places for values >= 1km
          const km = value / 1000
          return `${km.toFixed(2)} km`
        } else {
          // Show in meters with no decimals for values < 1km
          return `${Math.round(value)} m`
        }
      }
      computedFeatures.push({ label: 'Length:', value: formatLength(lengthM) })
    }
    
    if (affected !== undefined && !vulnerabilityAnalysisEnabled) {
      const affectedValue = typeof affected === 'boolean' ? (affected ? 'Yes' : 'No') : formatValue(affected)
      computedFeatures.push({ label: 'Affected:', value: affectedValue })
    }
    
    // For line segments, prioritize showing max and avg exposure levels
    // For points, show single exposure level
    if (exposureLevelMax !== undefined || exposureLevelAvg !== undefined) {
      // LineString segment feature - show max and avg exposure levels
      if (exposureLevelMax !== undefined && exposureLevelMax !== null) {
        computedFeatures.push({ label: 'Max Exposure Level:', value: formatExposureLevel(exposureLevelMax, selectedHazard?.unit) })
      }
      if (exposureLevelAvg !== undefined && exposureLevelAvg !== null) {
        computedFeatures.push({ label: 'Avg Exposure Level:', value: formatExposureLevel(exposureLevelAvg, selectedHazard?.unit) })
      }
    } else if (exposureLevel !== undefined && exposureLevel !== null) {
      // Point feature - show single exposure level
      computedFeatures.push({ label: 'Exposure Level:', value: formatExposureLevel(exposureLevel, selectedHazard?.unit) })
    }
    
    // Add vulnerability and damage cost if available
    if (vulnerability !== undefined && vulnerability !== null) {
      const formatVulnerability = (value: number): string => {
        // Format as percentage with 1-2 decimal places
        const percentage = value * 100
        return `${percentage.toFixed(percentage < 1 ? 2 : 1)}%`
      }
      computedFeatures.push({ label: 'Vulnerability:', value: formatVulnerability(vulnerability) })
    }
    
    if (damageCost !== undefined && damageCost !== null) {
      const formatDamageCost = (value: number): string => {
        return value.toLocaleString('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        })
      }
      computedFeatures.push({ label: 'Damage Cost:', value: formatDamageCost(damageCost) })
    }
    
    // Display all computed features at the top (bold font distinguishes them)
    if (computedFeatures.length > 0) {
      for (let i = 0; i < computedFeatures.length; i++) {
        const feature = computedFeatures[i]
        html += `<div class="flex justify-between text-xs py-0.5">`
        html += `<span class="font-semibold text-gray-800 mr-2">${feature.label}</span>`
        html += `<span class="text-gray-900 text-right flex-1 font-medium">${feature.value}</span>`
        html += `</div>`
      }
      // Add horizontal rule between computed features and other properties
      // Match spacing of title: pb-1.5 (space before border) + mb-1.5 (space after border)
      html += `<div class="pt-1.5 mb-1.5 border-b border-gray-200"></div>`
    }

    // Sort properties alphabetically
    // Exclude affected, exposure level, vulnerability, and damage_cost properties from the main list since they're shown at the top
    const sortedKeys = Object.keys(properties)
      .filter(key => !['affected', 'exposure_level', 'exposure_level_max', 'exposure_level_avg', 'length_m', 'vulnerability', 'damage_cost'].includes(key))
      .sort((a, b) => a.localeCompare(b))

    for (const key of sortedKeys) {
      const value = properties[key]
      const formattedKey = formatPropertyName(key)
      const formattedValue = formatValue(value)
      
      html += `<div class="flex justify-between text-xs py-0.5">`
      html += `<span class="font-medium text-gray-700 mr-2">${formattedKey}:</span>`
      html += `<span class="text-gray-900 text-right flex-1">${formattedValue}</span>`
      html += `</div>`
    }

    html += '</div></div>'
    return html
  }

  // Deterministic layer helpers for toggling
  const ensureStyleLoaded = (cb: () => void) => {
    const tryRun = () => {
      if (map.current?.isStyleLoaded()) cb()
      else setTimeout(tryRun, 50)
    }
    tryRun()
  }

  const removeHazard = () => {
    if (!map.current) return
    const hazardLayerId = 'hazard-raster-layer'
    const hazardSourceId = 'hazard-raster-source'
    if (map.current.getLayer(hazardLayerId)) map.current.removeLayer(hazardLayerId)
    if (map.current.getSource(hazardSourceId)) map.current.removeSource(hazardSourceId)
  }

  const addHazard = () => {
    if (!map.current) return
    const props = currentProps.current
    const selHazard = props.selectedHazard
    if (!selHazard) return
    const hazardLayerId = 'hazard-raster-layer'
    const hazardSourceId = 'hazard-raster-source'
    const tileUrl = `/api/tiles/${selHazard.id}/{z}/{x}/{y}.png?palette=${props.colorPalette}`
    removeHazard()
    try {
      map.current.addSource(hazardSourceId, { type: 'raster', tiles: [tileUrl], tileSize: 256 })
      const beforeLayer = map.current.getLayer('infrastructure-layer') ? 'infrastructure-layer' : undefined
      map.current.addLayer({ id: hazardLayerId, type: 'raster', source: hazardSourceId, paint: { 'raster-opacity': props.hazardOpacity / 100 } }, beforeLayer)
      map.current.setLayoutProperty(hazardLayerId, 'visibility', hazardVisible ? 'visible' : 'none')
    } catch (e) { console.error('addHazard error:', e) }
  }

  // Add popup click handlers to infrastructure layers
  const addPopupHandlers = (layerId: string) => {
    if (!map.current) return
    
    const handleClick = (e: maplibregl.MapLayerMouseEvent) => {
      if (!map.current || !e.features || e.features.length === 0) return
      
      const feature = e.features[0]
      const properties = feature.properties || {}
      const coordinates = (e.lngLat as any).toArray()
      
      // Close existing popup if any
      if (popup.current) {
        popup.current.remove()
      }
      
      // Create new popup
      popup.current = new maplibregl.Popup({
        closeButton: true,
        closeOnClick: true,
        maxWidth: '300px',
      })
        .setLngLat(coordinates)
        .setHTML(formatFeatureProperties(properties))
        .addTo(map.current)
    }
    
    // Use type assertion for layer-specific event handlers
    ;(map.current.on as any)('click', layerId, handleClick)
    
    // Change cursor on hover
    ;(map.current.on as any)('mouseenter', layerId, () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = 'pointer'
      }
    })
    
    ;(map.current.on as any)('mouseleave', layerId, () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = ''
      }
    })
  }

  // Remove popup click handlers from infrastructure layers
  const removePopupHandlers = (layerId: string) => {
    if (!map.current) return
    ;(map.current.off as any)('click', layerId)
    ;(map.current.off as any)('mouseenter', layerId)
    ;(map.current.off as any)('mouseleave', layerId)
  }

  const removeInfrastructure = () => {
    if (!map.current) return
    const sourceId = 'infrastructure-source'
    const layerId = 'infrastructure-layer'
    const affectedLayerId = 'infrastructure-affected'
    
    // Remove popup handlers before removing layers
    removePopupHandlers(layerId)
    removePopupHandlers(affectedLayerId)
    
    // Close popup if open
    if (popup.current) {
      popup.current.remove()
      popup.current = null
    }
    
    if (map.current.getLayer(affectedLayerId)) map.current.removeLayer(affectedLayerId)
    if (map.current.getLayer(layerId)) map.current.removeLayer(layerId)
    if (map.current.getSource(sourceId)) map.current.removeSource(sourceId)
  }

  const getInfraStyle = (isVulnerabilityMode: boolean, maxVuln = 1): { unaffectedColor: any; affectedColor: any; unaffectedFilter: any; affectedFilter: any } => {
    const sqrtMax = Math.sqrt(maxVuln)
    return {
      unaffectedColor: isVulnerabilityMode ? '#6b7280' : '#10b981',
      affectedColor: isVulnerabilityMode
        ? ['interpolate', ['linear'],
            ['sqrt', ['coalesce', ['get', 'vulnerability'], 0]],
            0, '#10b981',
            sqrtMax * 0.5, '#f59e0b',
            sqrtMax, '#ef4444']
        : '#ef4444',
      unaffectedFilter: isVulnerabilityMode
        ? ['boolean', false]
        : ['==', ['get', 'affected'], false],
      affectedFilter: isVulnerabilityMode
        ? ['boolean', true]
        : ['==', ['get', 'affected'], true],
    }
  }

  const addInfrastructure = () => {
    if (!map.current) return
    const props = currentProps.current
    const file = props.uploadedFile
    if (!file) return
    const sourceId = 'infrastructure-source'
    const layerId = 'infrastructure-layer'
    const affectedLayerId = 'infrastructure-affected'
    const geoJson: any = props.analysisResult?.infrastructure_features || file.geojson
    const isPoint = (props.analysisResult?.geometry_type || file.geometry_type) === 'Point'
    const hasAffected = !!props.analysisResult?.infrastructure_features
    const isVulnMode = !!props.vulnerabilityAnalysisEnabled && hasAffected
    const style = getInfraStyle(isVulnMode, props.maxVulnerability)
    if (!geoJson) return
    removeInfrastructure()
    try {
      map.current.addSource(sourceId, { type: 'geojson', data: geoJson })
      if (hasAffected) {
        const needsGradientLayer = isVulnMode
          || (isPoint ? (props.analysisResult?.summary?.affected_count || 0) > 0 : (props.analysisResult?.summary?.affected_meters || 0) > 0)
        if (isPoint) {
          map.current.addLayer({ id: layerId, type: 'circle', source: sourceId, filter: style.unaffectedFilter, paint: { 'circle-radius': 5, 'circle-color': style.unaffectedColor, 'circle-opacity': 0.8 } })
          if (needsGradientLayer) {
            map.current.addLayer({ id: affectedLayerId, type: 'circle', source: sourceId, filter: style.affectedFilter, paint: { 'circle-radius': 5, 'circle-color': style.affectedColor, 'circle-opacity': 0.8 } })
            map.current.moveLayer(affectedLayerId)
          }
          map.current.moveLayer(layerId)
        } else {
          map.current.addLayer({ id: layerId, type: 'line', source: sourceId, filter: style.unaffectedFilter, paint: { 'line-color': style.unaffectedColor, 'line-width': 3, 'line-opacity': 0.8 } })
          if (needsGradientLayer) {
            map.current.addLayer({ id: affectedLayerId, type: 'line', source: sourceId, filter: style.affectedFilter, paint: { 'line-color': style.affectedColor, 'line-width': 3, 'line-opacity': 0.8 } })
            map.current.moveLayer(affectedLayerId)
          }
          map.current.moveLayer(layerId)
        }
      } else {
        if (isPoint) map.current.addLayer({ id: layerId, type: 'circle', source: sourceId, paint: { 'circle-radius': 5, 'circle-color': '#6b7280', 'circle-opacity': 0.8 } })
        else map.current.addLayer({ id: layerId, type: 'line', source: sourceId, paint: { 'line-color': '#6b7280', 'line-width': 3, 'line-opacity': 0.8 } })
        // Move infrastructure layer to top to ensure it's above hazard
        map.current.moveLayer(layerId)
      }
      map.current.setLayoutProperty(layerId, 'visibility', infrastructureVisible ? 'visible' : 'none')
      if (map.current.getLayer(affectedLayerId)) map.current.setLayoutProperty(affectedLayerId, 'visibility', infrastructureVisible ? 'visible' : 'none')
      
      // Add popup handlers to both layers
      addPopupHandlers(layerId)
      if (map.current.getLayer(affectedLayerId)) {
        addPopupHandlers(affectedLayerId)
      }
    } catch (e) { console.error('addInfrastructure error:', e) }
  }

  // Update basemap - preserve existing layers
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    
    // Use current prop values from refs
    const props = currentProps.current
    
    // Check if we have any layers to preserve
    const hasInfrastructure = props.uploadedFile && (props.analysisResult?.infrastructure_features || props.uploadedFile.geojson)
    const hasHazard = props.selectedHazard !== null
    
    if (!hasInfrastructure && !hasHazard) {
      // No layers to preserve, just change basemap
      map.current.setStyle(basemapStyles[basemap])
      return
    }
    
    // Store what we need to restore in a closure-safe way
    const needsRestore = {
      infrastructure: hasInfrastructure,
      hazard: hasHazard,
      geoJson: props.analysisResult?.infrastructure_features || props.uploadedFile?.geojson || null,
      isPoint: props.analysisResult?.geometry_type === 'Point' || props.uploadedFile?.geometry_type === 'Point',
      hasAffected: !!(props.analysisResult?.infrastructure_features),
      hazardId: props.selectedHazard?.id,
      hazardOpacity: props.hazardOpacity,
      colorPalette: props.colorPalette,
      analysisResult: props.analysisResult,
      vulnerabilityAnalysisEnabled: !!props.vulnerabilityAnalysisEnabled,
      maxVulnerability: props.maxVulnerability,
    }
    
    // Function to restore layers
    const restoreLayers = () => {
      if (!map.current || !map.current.isStyleLoaded()) {
        setTimeout(restoreLayers, 100)
        return
      }
      
      // Restore hazard layer first if needed
      if (needsRestore.hazard && needsRestore.hazardId) {
        try {
          const hazardSourceId = 'hazard-raster-source'
          const hazardLayerId = 'hazard-raster-layer'
          const tileUrl = `/api/tiles/${needsRestore.hazardId}/{z}/{x}/{y}.png?palette=${needsRestore.colorPalette}`
          
          // Remove if exists
          if (map.current.getLayer(hazardLayerId)) {
            map.current.removeLayer(hazardLayerId)
          }
          if (map.current.getSource(hazardSourceId)) {
            map.current.removeSource(hazardSourceId)
          }
          
          // Add source and layer
          map.current.addSource(hazardSourceId, {
            type: 'raster',
            tiles: [tileUrl],
            tileSize: 256,
          })
          
          map.current.addLayer({
            id: hazardLayerId,
            type: 'raster',
            source: hazardSourceId,
            paint: {
              'raster-opacity': needsRestore.hazardOpacity / 100,
            },
          })
          
          // Apply visibility state
          if (!hazardVisible) {
            map.current.setLayoutProperty(hazardLayerId, 'visibility', 'none')
          }
        } catch (e) {
          console.error('Error restoring hazard layer:', e)
        }
      }
      
      // Restore infrastructure layer
      if (needsRestore.infrastructure && needsRestore.geoJson) {
        try {
          const sourceId = 'infrastructure-source'
          const layerId = 'infrastructure-layer'
          const affectedLayerId = 'infrastructure-affected'
          
          // Remove existing layers
          if (map.current.getLayer(affectedLayerId)) {
            map.current.removeLayer(affectedLayerId)
          }
          if (map.current.getLayer(layerId)) {
            map.current.removeLayer(layerId)
          }
          if (map.current.getSource(sourceId)) {
            map.current.removeSource(sourceId)
          }
          
          // Add source
          map.current.addSource(sourceId, {
            type: 'geojson',
            data: needsRestore.geoJson,
          })
          
          // Add layers based on whether we have affected status
          const isRestoreVulnMode = needsRestore.vulnerabilityAnalysisEnabled && needsRestore.hasAffected
          const restoreStyle = getInfraStyle(isRestoreVulnMode, needsRestore.maxVulnerability)
          if (needsRestore.hasAffected) {
            const needsGradientLayer = isRestoreVulnMode
              || (needsRestore.isPoint
                ? (needsRestore.analysisResult?.summary?.affected_count || 0) > 0
                : (needsRestore.analysisResult?.summary?.affected_meters || 0) > 0)
            if (needsRestore.isPoint) {
                map.current.addLayer({
                  id: layerId,
                  type: 'circle',
                  source: sourceId,
                  filter: restoreStyle.unaffectedFilter,
                  paint: {
                    'circle-radius': 5,
                    'circle-color': restoreStyle.unaffectedColor,
                    'circle-opacity': 0.8,
                  },
                })
                
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(layerId, 'visibility', 'none')
                }
                
                if (needsGradientLayer) {
                  map.current.addLayer({
                    id: affectedLayerId,
                    type: 'circle',
                    source: sourceId,
                    filter: restoreStyle.affectedFilter,
                    paint: {
                      'circle-radius': 5,
                      'circle-color': restoreStyle.affectedColor,
                      'circle-opacity': 0.8,
                    },
                  })
                  
                  if (!infrastructureVisible) {
                    map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                  }
                }
            } else {
              map.current.addLayer({
                id: layerId,
                type: 'line',
                source: sourceId,
                filter: restoreStyle.unaffectedFilter,
                paint: {
                  'line-color': restoreStyle.unaffectedColor,
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
              
              if (needsGradientLayer) {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'line',
                  source: sourceId,
                  filter: restoreStyle.affectedFilter,
                  paint: {
                    'line-color': restoreStyle.affectedColor,
                    'line-width': 3,
                    'line-opacity': 0.8,
                  },
                })
                
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              }
              
              addPopupHandlers(layerId)
              if (map.current.getLayer(affectedLayerId)) {
                addPopupHandlers(affectedLayerId)
              }
            }
          } else {
            // No analysis yet - show all features in dark grey
            if (needsRestore.isPoint) {
              map.current.addLayer({
                id: layerId,
                type: 'circle',
                source: sourceId,
                paint: {
                  'circle-radius': 5,
                  'circle-color': '#6b7280',
                  'circle-opacity': 0.8,
                },
              })
              
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            } else {
              map.current.addLayer({
                id: layerId,
                type: 'line',
                source: sourceId,
                paint: {
                  'line-color': '#6b7280',
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            }
            
            // Add popup handlers
            addPopupHandlers(layerId)
            if (map.current.getLayer(affectedLayerId)) {
              addPopupHandlers(affectedLayerId)
            }
          }
        } catch (e) {
          console.error('Error restoring infrastructure layer:', e)
        }
      }
      
      // Apply visibility states after restoration
      setTimeout(() => {
        if (map.current) {
          applyInfrastructureVisibility(infrastructureVisible)
          applyHazardVisibility(hazardVisible)
        }
      }, 150)
      
      isRestoringLayers.current = false
    }
    
    isRestoringLayers.current = true
    
    // Listen for style load
    const handleStyleLoad = () => {
      setTimeout(restoreLayers, 100)
    }
    
    map.current.once('style.load', handleStyleLoad)
    
    // Also try after a delay in case the event doesn't fire
    const timeoutId = setTimeout(() => {
      if (isRestoringLayers.current) {
        restoreLayers()
      }
    }, 1000)
    
    // Change the style
    map.current.setStyle(basemapStyles[basemap])
    
    // Cleanup timeout if style.load fires
    map.current.once('style.load', () => {
      clearTimeout(timeoutId)
    })
  }, [basemap, mapLoaded])

  // Reset bounds tracking when file changes
  useEffect(() => {
    if (uploadedFile) {
      // Reset bounds tracking when a new file is uploaded
      const fileId = uploadedFile.file_id
      if (boundsSetForFile.current !== fileId) {
        lastBounds.current = null
      }
    } else {
      // Clear tracking when file is removed
      boundsSetForFile.current = null
      lastBounds.current = null
    }
  }, [uploadedFile?.file_id])

  // Add/update hazard raster layer
  useEffect(() => {
    // Skip if we're restoring layers after basemap change
    if (isRestoringLayers.current) return
    
    if (!map.current || !mapLoaded || !selectedHazard) {
      // Remove hazard layer if hazard is deselected
      if (map.current) {
        const hazardLayerId = 'hazard-raster-layer'
        const hazardSourceId = 'hazard-raster-source'
        if (map.current.getLayer(hazardLayerId)) {
          map.current.removeLayer(hazardLayerId)
        }
        if (map.current.getSource(hazardSourceId)) {
          map.current.removeSource(hazardSourceId)
        }
      }
      return
    }

    const hazardSourceId = 'hazard-raster-source'
    const hazardLayerId = 'hazard-raster-layer'

    // Debounce hazard layer updates to prevent rapid successive reloads
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
    }
    
    debounceTimer.current = setTimeout(() => {
      const waitForStyleAndAddHazard = () => {
        if (!map.current?.isStyleLoaded()) {
          setTimeout(waitForStyleAndAddHazard, 100)
          return
        }

        // Check if layer already exists - if so, just update it
        const existingLayer = map.current.getLayer(hazardLayerId)
        const existingSource = map.current.getSource(hazardSourceId)
        
        if (existingLayer && existingSource) {
          // Update existing layer properties (opacity and tile URL if palette changed)
          try {
            map.current.setPaintProperty(hazardLayerId, 'raster-opacity', hazardOpacity / 100)
            
            // Update source if palette changed (remove and re-add source)
            const currentTiles = (existingSource as any).tiles
            const newTileUrl = `/api/tiles/${selectedHazard.id}/{z}/{x}/{y}.png?palette=${colorPalette}`
            if (currentTiles && currentTiles[0] !== newTileUrl) {
              map.current.removeLayer(hazardLayerId)
              map.current.removeSource(hazardSourceId)
              
              map.current.addSource(hazardSourceId, {
                type: 'raster',
                tiles: [newTileUrl],
                tileSize: 256,
              })
              
              const beforeLayer = map.current.getLayer('infrastructure-layer') ? 'infrastructure-layer' : undefined
              map.current.addLayer({
                id: hazardLayerId,
                type: 'raster',
                source: hazardSourceId,
                paint: {
                  'raster-opacity': hazardOpacity / 100,
                },
              }, beforeLayer)
              
              // Apply visibility state
              if (!hazardVisible) {
                map.current.setLayoutProperty(hazardLayerId, 'visibility', 'none')
              }
            } else {
              // Just update visibility if layer exists
              if (!hazardVisible) {
                map.current.setLayoutProperty(hazardLayerId, 'visibility', 'none')
              } else {
                map.current.setLayoutProperty(hazardLayerId, 'visibility', 'visible')
              }
            }
            return
          } catch (error) {
            console.error('Error updating hazard layer:', error)
            // Fall through to add new layer
          }
        }

        // Remove existing hazard layer if updating
        if (map.current.getLayer(hazardLayerId)) {
          map.current.removeLayer(hazardLayerId)
        }
        if (map.current.getSource(hazardSourceId)) {
          map.current.removeSource(hazardSourceId)
        }

        // Add hazard raster layer using backend tile service
        try {
          // Use backend tile endpoint
          const tileUrl = `/api/tiles/${selectedHazard.id}/{z}/{x}/{y}.png?palette=${colorPalette}`
          
          map.current.addSource(hazardSourceId, {
            type: 'raster',
            tiles: [tileUrl],
            tileSize: 256,
          })

          // Add hazard layer - it should be below infrastructure
          // We'll ensure infrastructure is moved above it after adding
          map.current.addLayer({
            id: hazardLayerId,
            type: 'raster',
            source: hazardSourceId,
            paint: {
              'raster-opacity': hazardOpacity / 100,
            },
          })
          
          // If infrastructure layers exist, ensure they're above the hazard layer
          // Move them in the correct order: infrastructure (green) first, then affected (red) on top
          const infrastructureLayer = map.current.getLayer('infrastructure-layer')
          const affectedLayer = map.current.getLayer('infrastructure-affected')
          if (infrastructureLayer) {
            // Move infrastructure layer (green/unaffected) to top first
            map.current.moveLayer('infrastructure-layer')
          }
          if (affectedLayer) {
            // Move affected layer (red) to top last, so it renders above infrastructure
            map.current.moveLayer('infrastructure-affected')
          }
          
          // Apply visibility state
          if (!hazardVisible) {
            map.current.setLayoutProperty(hazardLayerId, 'visibility', 'none')
          }
        } catch (error) {
          console.error('Error adding hazard layer:', error)
        }
      }

      waitForStyleAndAddHazard()
    }, 100) // Debounce delay
    
    // Cleanup function
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
      }
    }
  }, [selectedHazard, hazardOpacity, colorPalette, mapLoaded, hazardVisible])

  // Add/update infrastructure layer with affected/unaffected styling
  useEffect(() => {
    // Skip if we're restoring layers after basemap change
    if (isRestoringLayers.current) return
    
    if (!map.current || !mapLoaded || !uploadedFile) {
      // Remove infrastructure layers if file is removed
      if (map.current) {
        const sourceId = 'infrastructure-source'
        const layerId = 'infrastructure-layer'
        const affectedLayerId = 'infrastructure-affected'
        
        // Remove popup handlers
        removePopupHandlers(layerId)
        removePopupHandlers(affectedLayerId)
        
        // Close popup if open
        if (popup.current) {
          popup.current.remove()
          popup.current = null
        }
        
        if (map.current.getLayer(affectedLayerId)) {
          map.current.removeLayer(affectedLayerId)
        }
        if (map.current.getLayer(layerId)) {
          map.current.removeLayer(layerId)
        }
        if (map.current.getSource(sourceId)) {
          map.current.removeSource(sourceId)
        }
      }
      return
    }

    const sourceId = 'infrastructure-source'
    const layerId = 'infrastructure-layer'
    const affectedLayerId = 'infrastructure-affected'

    const waitForStyle = () => {
      if (!map.current?.isStyleLoaded()) {
        setTimeout(waitForStyle, 100)
        return
      }

      // Check if we need to update or just add
      const existingSource = map.current.getSource(sourceId)
      const existingLayer = map.current.getLayer(layerId)
      const existingAffectedLayer = map.current.getLayer(affectedLayerId)
      
      // Determine which GeoJSON to use
      let geoJson = null
      let isPoint = false
      let needsUpdate = false
      
      if (analysisResult?.infrastructure_features) {
        // Use analysis result if available (has affected status)
        geoJson = analysisResult.infrastructure_features
        isPoint = analysisResult.geometry_type === 'Point'
        
        // Always update if we don't have the affected layer (switching from plain to affected/unaffected styling)
        // This is critical - we need to show affected/unaffected coloring, not plain styling
        if (existingLayer && !existingAffectedLayer) {
          needsUpdate = true // Force update to show affected/unaffected coloring
        } else if (!existingLayer) {
          // No layer exists yet, need to create it
          needsUpdate = true
        } else {
          // Even if layers exist, check if source data needs updating
          if (existingSource && (existingSource as any)._data) {
            const currentData = JSON.stringify((existingSource as any)._data)
            const newData = JSON.stringify(geoJson)
            needsUpdate = currentData !== newData
            // If data is the same but we have analysis results, we still need to update
            // because the layer styling might need to change
            if (!needsUpdate && !existingAffectedLayer) {
              needsUpdate = true
            }
          } else {
            needsUpdate = true
          }
        }
      } else if (uploadedFile.geojson) {
        // Use uploaded file GeoJSON for initial display
        geoJson = uploadedFile.geojson
        isPoint = uploadedFile.geometry_type === 'Point'
        
        // Check if source data needs updating
        if (existingSource && (existingSource as any)._data) {
          const currentData = JSON.stringify((existingSource as any)._data)
          const newData = JSON.stringify(geoJson)
          needsUpdate = currentData !== newData
        } else {
          needsUpdate = true
        }
        
        // Always update if we're switching from analysis results to plain styling
        if (!needsUpdate && existingAffectedLayer) {
          needsUpdate = true // Switching from affected/unaffected to plain styling
        }
      }

      // If source exists and data hasn't changed and we're not switching between analysis/plain modes, no update needed
      // BUT: always update if we have analysis results and don't have the affected layer yet
      if (existingSource && existingLayer && !needsUpdate) {
        // Double-check: if we have analysis results but no affected layer, we need to update
        if (analysisResult?.infrastructure_features && !existingAffectedLayer) {
          needsUpdate = true
        } else if (!analysisResult?.infrastructure_features && existingAffectedLayer) {
          // Switching from analysis results back to plain - need to update
          needsUpdate = true
        } else {
          return // No update needed
        }
      }

      // Remove existing layers only if we're updating
      if (needsUpdate && geoJson) {
        // Remove popup handlers before removing layers
        removePopupHandlers(layerId)
        removePopupHandlers(affectedLayerId)
        
        // Close popup if open
        if (popup.current) {
          popup.current.remove()
          popup.current = null
        }
        
        if (map.current.getLayer(affectedLayerId)) {
          map.current.removeLayer(affectedLayerId)
        }
        if (map.current.getLayer(layerId)) {
          map.current.removeLayer(layerId)
        }
        if (map.current.getSource(sourceId)) {
          map.current.removeSource(sourceId)
        }
      } else if (!geoJson) {
        // No GeoJSON available
        return
      }

      if (geoJson && needsUpdate) {
        try {
          // Add source (we already removed it above if it existed)
          map.current.addSource(sourceId, {
            type: 'geojson',
            data: geoJson,
          })

          // If we have analysis results, show affected/unaffected coloring
          if (analysisResult?.infrastructure_features) {
            const mainStyle = getInfraStyle(vulnerabilityAnalysisEnabled, maxVulnerability)
            const needsGradientLayer = vulnerabilityAnalysisEnabled
              || (isPoint
                ? (analysisResult.summary.affected_count || 0) > 0
                : (analysisResult.summary.affected_meters || 0) > 0)
            if (isPoint) {
              map.current.addLayer({
                id: layerId,
                type: 'circle',
                source: sourceId,
                filter: mainStyle.unaffectedFilter,
                paint: {
                  'circle-radius': 5,
                  'circle-color': mainStyle.unaffectedColor,
                  'circle-opacity': 0.8,
                },
              })
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            } else {
              map.current.addLayer({
                id: layerId,
                type: 'line',
                source: sourceId,
                filter: mainStyle.unaffectedFilter,
                paint: {
                  'line-color': mainStyle.unaffectedColor,
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            }

            if (needsGradientLayer) {
              if (isPoint) {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'circle',
                  source: sourceId,
                  filter: mainStyle.affectedFilter,
                  paint: {
                    'circle-radius': 5,
                    'circle-color': mainStyle.affectedColor,
                    'circle-opacity': 0.8,
                  },
                })
                
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              } else {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'line',
                  source: sourceId,
                  filter: mainStyle.affectedFilter,
                  paint: {
                    'line-color': mainStyle.affectedColor,
                    'line-width': 3,
                    'line-opacity': 0.8,
                  },
                })
                
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              }
              
              addPopupHandlers(layerId)
              if (map.current.getLayer(affectedLayerId)) {
                addPopupHandlers(affectedLayerId)
              }
            }
            
            map.current.moveLayer(layerId)
            if (map.current.getLayer(affectedLayerId)) {
              map.current.moveLayer(affectedLayerId)
            }
          } else {
            // No analysis yet - show all features in dark grey
            if (isPoint) {
              // Add layer without beforeId so it goes on top, then move it after hazard if hazard exists
              map.current.addLayer({
                id: layerId,
                type: 'circle',
                source: sourceId,
                paint: {
                  'circle-radius': 5,
                  'circle-color': '#6b7280', // dark grey
                  'circle-opacity': 0.8,
                },
              })
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            } else {
              map.current.addLayer({
                id: layerId,
                type: 'line',
                source: sourceId,
                paint: {
                  'line-color': '#6b7280', // dark grey
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            }
            
            // Ensure infrastructure is above hazard
            map.current.moveLayer(layerId)
            
            // Add popup handlers
            addPopupHandlers(layerId)
          }

          // Only fit bounds if we haven't set them for this file yet, or if bounds changed
          const bounds = uploadedFile.bounds
          const boundsKey = `${bounds.minx},${bounds.miny},${bounds.maxx},${bounds.maxy}`
          const fileId = uploadedFile.file_id
          
          // Only call fitBounds if:
          // 1. We haven't set bounds for this file yet, OR
          // 2. The bounds have changed
          if (boundsSetForFile.current !== fileId || lastBounds.current !== boundsKey) {
            const bbox: [number, number, number, number] = [
              bounds.minx,
              bounds.miny,
              bounds.maxx,
              bounds.maxy
            ]
            // Debounce fitBounds to avoid rapid successive calls
            if (debounceTimer.current) {
              clearTimeout(debounceTimer.current)
            }
            debounceTimer.current = setTimeout(() => {
              if (map.current) {
                map.current.fitBounds(bbox, { padding: 50, duration: 0 }) // duration: 0 prevents animation that can trigger tile reloads
                boundsSetForFile.current = fileId
                lastBounds.current = boundsKey
              }
            }, 75)
          }
        } catch (error) {
          console.error('Error adding infrastructure layer:', error)
          // Only fit bounds if not already set for this file
          const bounds = uploadedFile.bounds
          const boundsKey = `${bounds.minx},${bounds.miny},${bounds.maxx},${bounds.maxy}`
          const fileId = uploadedFile.file_id
          
          if (boundsSetForFile.current !== fileId || lastBounds.current !== boundsKey) {
            const bbox: [number, number, number, number] = [
              bounds.minx,
              bounds.miny,
              bounds.maxx,
              bounds.maxy
            ]
            if (debounceTimer.current) {
              clearTimeout(debounceTimer.current)
            }
            debounceTimer.current = setTimeout(() => {
              if (map.current) {
                map.current.fitBounds(bbox, { padding: 50, duration: 0 })
                boundsSetForFile.current = fileId
                lastBounds.current = boundsKey
              }
            }, 100)
          }
        }
      } else {
        // Only fit bounds if no GeoJSON available yet and bounds not set
        const bounds = uploadedFile.bounds
        const boundsKey = `${bounds.minx},${bounds.miny},${bounds.maxx},${bounds.maxy}`
        const fileId = uploadedFile.file_id
        
        if (boundsSetForFile.current !== fileId || lastBounds.current !== boundsKey) {
          const bbox: [number, number, number, number] = [
            bounds.minx,
            bounds.miny,
            bounds.maxx,
            bounds.maxy
          ]
          if (debounceTimer.current) {
            clearTimeout(debounceTimer.current)
          }
          debounceTimer.current = setTimeout(() => {
            if (map.current) {
              map.current.fitBounds(bbox, { padding: 50, duration: 0 })
              boundsSetForFile.current = fileId
              lastBounds.current = boundsKey
            }
          }, 100)
        }
      }
    }

    waitForStyle()
    
    // Cleanup function
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
      }
    }
  }, [uploadedFile, analysisResult, mapLoaded, infrastructureVisible, vulnerabilityAnalysisEnabled, maxVulnerability])

  // Basemap style switcher (always on)
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    
    // Use current prop values from refs
    const props = currentProps.current
    const hasInfrastructure = props.uploadedFile && (props.analysisResult?.infrastructure_features || props.uploadedFile.geojson)
    const hasHazard = props.selectedHazard !== null
    
    if (!hasInfrastructure && !hasHazard) {
      map.current.setStyle(basemapStyles[basemap])
      return
    }
    
    // Preserve layers similar to earlier logic
    isRestoringLayers.current = true
    const needsRestore = {
      infrastructure: hasInfrastructure,
      hazard: hasHazard,
    }
    
    const restoreAfterStyle = () => {
      setTimeout(() => {
        if (!map.current) {
          isRestoringLayers.current = false
          return
        }
        if (needsRestore.hazard && currentProps.current.selectedHazard) addHazard()
        if (needsRestore.infrastructure && currentProps.current.uploadedFile) addInfrastructure()
        isRestoringLayers.current = false
      }, 100)
    }
    
    map.current.once('style.load', restoreAfterStyle)
    map.current.setStyle(basemapStyles[basemap])
  }, [basemap, mapLoaded])

  // Helper function to apply infrastructure visibility
  const applyInfrastructureVisibility = (visible: boolean) => {
    if (!map.current) return
    
    const layerId = 'infrastructure-layer'
    const affectedLayerId = 'infrastructure-affected'
    
    try {
      if (map.current.getLayer(layerId)) {
        map.current.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none')
      }
      if (map.current.getLayer(affectedLayerId)) {
        map.current.setLayoutProperty(affectedLayerId, 'visibility', visible ? 'visible' : 'none')
      }
    } catch (error) {
      console.error('Error applying infrastructure visibility:', error)
    }
  }

  // Toggle infrastructure layer visibility
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    ensureStyleLoaded(() => {
      if (!map.current) return
      if (!infrastructureVisible) {
        removeInfrastructure()
      } else {
        // Only call addInfrastructure if we don't have layers yet
        // The main infrastructure useEffect handles updates when analysisResult changes
        const layerId = 'infrastructure-layer'
        if (uploadedFile && !map.current.getLayer(layerId)) {
          addInfrastructure()
        }
      }
    })
  }, [infrastructureVisible, mapLoaded, uploadedFile])

  // Helper function to apply hazard visibility
  const applyHazardVisibility = (visible: boolean) => {
    if (!map.current) return
    
    const hazardLayerId = 'hazard-raster-layer'
    
    try {
      if (map.current.getLayer(hazardLayerId)) {
        map.current.setLayoutProperty(hazardLayerId, 'visibility', visible ? 'visible' : 'none')
      }
    } catch (error) {
      console.error('Error applying hazard visibility:', error)
    }
  }

  // Toggle hazard layer visibility
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    ensureStyleLoaded(() => {
      if (!hazardVisible) {
        removeHazard()
      } else {
        if (selectedHazard) addHazard()
      }
    })
  }, [hazardVisible, mapLoaded, selectedHazard, colorPalette, hazardOpacity])

  // Reset visibility when layers are added/removed
  useEffect(() => {
    if (uploadedFile) {
      setInfrastructureVisible(true)
    }
  }, [uploadedFile])

  useEffect(() => {
    if (selectedHazard) {
      setHazardVisible(true)
    }
  }, [selectedHazard])

  return (
    <div className="relative flex-1 h-full m-0 p-0">
      {/* Basemap Selector */}
      <div className="absolute top-4 right-4 z-10 bg-white/80 rounded-lg px-1">
        <Select
          value={basemap}
          onChange={(e) => onBasemapChange(e.target.value as Basemap)}
          className="w-40 border-0 bg-transparent"
        >
          <option value="positron">CartoDB Positron</option>
          <option value="dark-matter">CartoDB Dark Matter</option>
          <option value="osm">OpenStreetMap</option>
          <option value="topo">OpenTopoMap</option>
          <option value="esri-street">Esri World Street Map</option>
          <option value="esri-topo">Esri World Topo Map</option>
          <option value="esri-terrain">Esri World Terrain</option>
          <option value="esri-ocean">Esri Ocean Basemap</option>
          <option value="esri-imagery">Esri World Imagery</option>
          <option value="google-maps">Google Maps</option>
          <option value="google-terrain">Google Terrain</option>
          <option value="google-hybrid">Google Hybrid</option>
          <option value="google-satellite">Google Satellite</option>
        </Select>
      </div>

      {/* Layer Control Panel (only show if at least one layer is available) */}
      {(uploadedFile || selectedHazard) && (
        <div className="absolute top-16 right-4 z-10 bg-white/80 rounded-lg px-3 py-2 space-y-2">
          {/* Infrastructure checkbox - only if infrastructure is uploaded */}
          {uploadedFile && (
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={infrastructureVisible}
                onChange={(e) => setInfrastructureVisible(e.target.checked)}
                className="w-4 h-4 text-gray-600 rounded"
              />
              <span className="text-sm text-gray-700">Infrastructure</span>
            </label>
          )}

          {/* Hazard layer checkbox - only if hazard is selected */}
          {selectedHazard && (
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={hazardVisible}
                onChange={(e) => setHazardVisible(e.target.checked)}
                className="w-4 h-4 text-gray-600 rounded"
              />
              <span className="text-sm text-gray-700">Hazard Layer</span>
            </label>
          )}
        </div>
      )}

      {/* Hazard Color Bar Legend */}
      {selectedHazard && hazardStats && hazardVisible && (() => {
        const sqrtMin = Math.sqrt(hazardStats.min)
        const sqrtMax = Math.sqrt(hazardStats.max)
        const ticks = [0, 0.25, 0.5, 0.75, 1].map(p => {
          const val = Math.pow(p * (sqrtMax - sqrtMin) + sqrtMin, 2)
          return { position: p * 100, label: val >= 100 ? Math.round(val).toLocaleString() : val >= 1 ? val.toFixed(1) : val.toFixed(2) }
        })
        const stops = COLORMAP_STOPS[colorPalette]
        const gradient = `linear-gradient(to right, ${stops.map((c, i) => `${c} ${(i / (stops.length - 1) * 100).toFixed(0)}%`).join(', ')})`

        // Build legend title with hazard unit, avoiding double parentheses like ((index))
        let unitLabel: string | null = null
        if (selectedHazard.unit) {
          let u = selectedHazard.unit.trim()
          if (u.startsWith('(') && u.endsWith(')') && u.length > 2) {
            u = u.slice(1, -1).trim()
          }
          unitLabel = u || null
        }
        const legendTitle = unitLabel ? `Hazard Intensity (${unitLabel})` : 'Hazard Intensity'

        return (
          <div className="absolute top-4 left-4 z-10 bg-white/50 rounded-lg px-3 py-2">
            <div className="text-[10px] text-gray-800 mb-1">{legendTitle}</div>
            <div className="rounded" style={{ width: 220, height: 12, background: gradient }} />
            <div className="relative" style={{ width: 220, height: 16 }}>
              {ticks.map((t, i) => (
                <span
                  key={i}
                  className="absolute text-[10px] text-gray-800"
                  style={{
                    left: `${t.position}%`,
                    transform: i === ticks.length - 1 ? 'translateX(-100%)' : i === 0 ? 'none' : 'translateX(-50%)',
                    top: 2,
                  }}
                >
                  {t.label}
                </span>
              ))}
            </div>
          </div>
        )
      })()}

      {/* Vulnerability Color Bar Legend */}
      {vulnerabilityAnalysisEnabled && infrastructureVisible && analysisResult?.infrastructure_features && (() => {
        const maxPct = maxVulnerability * 100
        const ticks = [0, 0.25, 0.5, 0.75, 1].map(p => {
          const pct = Math.pow(p, 2) * maxPct
          return { position: p * 100, label: pct >= 10 ? `${Math.round(pct)}%` : pct >= 0.1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(2)}%` }
        })
        const gradient = 'linear-gradient(to right, #10b981 0%, #f59e0b 50%, #ef4444 100%)'
        return (
          <div className="absolute bottom-4 left-4 z-10 bg-white/50 rounded-lg px-3 py-2">
            <div className="text-[10px] text-gray-800 mb-1">Vulnerability</div>
            <div className="rounded" style={{ width: 220, height: 12, background: gradient }} />
            <div className="relative" style={{ width: 220, height: 16 }}>
              {ticks.map((t, i) => (
                <span
                  key={i}
                  className="absolute text-[10px] text-gray-800"
                  style={{
                    left: `${t.position}%`,
                    transform: i === ticks.length - 1 ? 'translateX(-100%)' : i === 0 ? 'none' : 'translateX(-50%)',
                    top: 2,
                  }}
                >
                  {t.label}
                </span>
              ))}
            </div>
          </div>
        )
      })()}

      {/* Map Container */}
      <div 
        ref={mapContainer} 
        className="w-full h-full" 
        style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
      />
    </div>
  )
}

