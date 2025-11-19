import { useEffect, useRef, useState } from 'react'
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
  loading?: boolean
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
  }
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
  loading = false,
}: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const isRestoringLayers = useRef(false)
  const popup = useRef<maplibregl.Popup | null>(null)
  
  // Layer visibility state (basemap always on)
  const [infrastructureVisible, setInfrastructureVisible] = useState(true)
  const [hazardVisible, setHazardVisible] = useState(true)
  
  // Store current prop values for restoration
  const currentProps = useRef({ uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette })
  
  // Track zoom level and bounds to prevent unnecessary reloads
  const lastZoomLevel = useRef<number | null>(null)
  const lastBounds = useRef<string | null>(null)
  const boundsSetForFile = useRef<string | null>(null)
  const debounceTimer = useRef<NodeJS.Timeout | null>(null)
  
  // Update refs when props change
  useEffect(() => {
    currentProps.current = { uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette }
  }, [uploadedFile, selectedHazard, analysisResult, hazardOpacity, colorPalette])

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
      })

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
          return value.toLocaleString()
        }
        // For decimals, show up to 6 decimal places but remove trailing zeros
        const formatted = value.toFixed(6).replace(/\.?0+$/, '')
        return parseFloat(formatted).toLocaleString()
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

    const formatExposureLevel = (value: any): string => {
      if (value === null || value === undefined) {
        return '<span class="text-gray-400 italic">N/A</span>'
      }
      if (typeof value === 'number') {
        // Format exposure levels with 2-4 decimal places
        // Show 4 decimal places, but remove trailing zeros
        const formatted = value.toFixed(4).replace(/\.?0+$/, '')
        return parseFloat(formatted).toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits: 4
        })
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
    const affected = properties['affected']
    const exposureLevel = properties['exposure_level']
    const exposureLevelMax = properties['exposure_level_max']
    const exposureLevelAvg = properties['exposure_level_avg']
    
    // Build list of computed features to display in a single shaded box
    const computedFeatures: Array<{label: string, value: string}> = []
    
    if (affected !== undefined) {
      const affectedValue = typeof affected === 'boolean' ? (affected ? 'Yes' : 'No') : formatValue(affected)
      computedFeatures.push({ label: 'Affected:', value: affectedValue })
    }
    
    if (exposureLevel !== undefined && exposureLevel !== null) {
      // Point feature - show single exposure level
      computedFeatures.push({ label: 'Exposure Level:', value: formatExposureLevel(exposureLevel) })
    } else if (exposureLevelMax !== undefined || exposureLevelAvg !== undefined) {
      // LineString feature - show max and avg exposure levels
      if (exposureLevelMax !== undefined && exposureLevelMax !== null) {
        computedFeatures.push({ label: 'Max Exposure Level:', value: formatExposureLevel(exposureLevelMax) })
      }
      if (exposureLevelAvg !== undefined && exposureLevelAvg !== null) {
        computedFeatures.push({ label: 'Avg Exposure Level:', value: formatExposureLevel(exposureLevelAvg) })
      }
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
    // Exclude affected and exposure level properties from the main list since they're shown at the top
    const sortedKeys = Object.keys(properties)
      .filter(key => !['affected', 'exposure_level', 'exposure_level_max', 'exposure_level_avg'].includes(key))
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
    if (!geoJson) return
    removeInfrastructure()
    try {
      map.current.addSource(sourceId, { type: 'geojson', data: geoJson })
      if (hasAffected) {
        if (isPoint) {
          map.current.addLayer({ id: layerId, type: 'circle', source: sourceId, filter: ['==', ['get', 'affected'], false], paint: { 'circle-radius': 5, 'circle-color': '#10b981', 'circle-opacity': 0.8 } })
          if ((props.analysisResult?.summary?.affected_count || 0) > 0) {
            map.current.addLayer({ id: affectedLayerId, type: 'circle', source: sourceId, filter: ['==', ['get', 'affected'], true], paint: { 'circle-radius': 5, 'circle-color': '#ef4444', 'circle-opacity': 0.8 } })
            // Move affected layer to top
            map.current.moveLayer(affectedLayerId)
          }
          // Move infrastructure layer to top to ensure it's above hazard
          map.current.moveLayer(layerId)
        } else {
          map.current.addLayer({ id: layerId, type: 'line', source: sourceId, filter: ['==', ['get', 'affected'], false], paint: { 'line-color': '#10b981', 'line-width': 3, 'line-opacity': 0.8 } })
          if ((props.analysisResult?.summary?.affected_meters || 0) > 0) {
            map.current.addLayer({ id: affectedLayerId, type: 'line', source: sourceId, filter: ['==', ['get', 'affected'], true], paint: { 'line-color': '#ef4444', 'line-width': 3, 'line-opacity': 0.8 } })
            // Move affected layer to top
            map.current.moveLayer(affectedLayerId)
          }
          // Move infrastructure layer to top to ensure it's above hazard
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
      // Store the GeoJSON to use - make sure we have a reference
      geoJson: props.analysisResult?.infrastructure_features || props.uploadedFile?.geojson || null,
      isPoint: props.analysisResult?.geometry_type === 'Point' || props.uploadedFile?.geometry_type === 'Point',
      hasAffected: !!(props.analysisResult?.infrastructure_features),
      hazardId: props.selectedHazard?.id,
      hazardOpacity: props.hazardOpacity,
      colorPalette: props.colorPalette,
      analysisResult: props.analysisResult
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
          if (needsRestore.hasAffected) {
            // Show affected/unaffected coloring
            if (needsRestore.isPoint) {
                map.current.addLayer({
                  id: layerId,
                  type: 'circle',
                  source: sourceId,
                  filter: ['==', ['get', 'affected'], false],
                  paint: {
                    'circle-radius': 5,
                    'circle-color': '#10b981',
                    'circle-opacity': 0.8,
                  },
                })
                
                // Apply visibility state
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(layerId, 'visibility', 'none')
                }
                
                // Check if there are affected features
                const affectedCount = (needsRestore.analysisResult?.summary?.affected_count || 0)
                if (affectedCount > 0) {
                  map.current.addLayer({
                    id: affectedLayerId,
                    type: 'circle',
                    source: sourceId,
                    filter: ['==', ['get', 'affected'], true],
                    paint: {
                      'circle-radius': 5,
                      'circle-color': '#ef4444',
                      'circle-opacity': 0.8,
                    },
                  })
                  
                  // Apply visibility state
                  if (!infrastructureVisible) {
                    map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                  }
                }
            } else {
              map.current.addLayer({
                id: layerId,
                type: 'line',
                source: sourceId,
                filter: ['==', ['get', 'affected'], false],
                paint: {
                  'line-color': '#10b981',
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
              
              const affectedMeters = (needsRestore.analysisResult?.summary?.affected_meters || 0)
              if (affectedMeters > 0) {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'line',
                  source: sourceId,
                  filter: ['==', ['get', 'affected'], true],
                  paint: {
                    'line-color': '#ef4444',
                    'line-width': 3,
                    'line-opacity': 0.8,
                  },
                })
                
                // Apply visibility state
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              }
              
              // Add popup handlers
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
            // Add unaffected features layer (green) - always above hazard layer
            if (isPoint) {
              // Add layer without beforeId so it goes on top, then move it after hazard if hazard exists
              map.current.addLayer({
                id: layerId,
                type: 'circle',
                source: sourceId,
                filter: ['==', ['get', 'affected'], false],
                paint: {
                  'circle-radius': 5,
                  'circle-color': '#10b981', // green for unaffected
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
                filter: ['==', ['get', 'affected'], false],
                paint: {
                  'line-color': '#10b981',
                  'line-width': 3,
                  'line-opacity': 0.8,
                },
              })
              // Apply visibility state
              if (!infrastructureVisible) {
                map.current.setLayoutProperty(layerId, 'visibility', 'none')
              }
            }

            // Add affected features layer (red)
            const affectedCount = isPoint 
              ? (analysisResult.summary.affected_count || 0)
              : (analysisResult.summary.affected_meters || 0) > 0 ? 1 : 0
            
            if (affectedCount > 0) {
              if (isPoint) {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'circle',
                  source: sourceId,
                  filter: ['==', ['get', 'affected'], true],
                  paint: {
                    'circle-radius': 5,
                    'circle-color': '#ef4444', // red for affected
                    'circle-opacity': 0.8,
                  },
                })
                
                // Apply visibility state
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              } else {
                map.current.addLayer({
                  id: affectedLayerId,
                  type: 'line',
                  source: sourceId,
                  filter: ['==', ['get', 'affected'], true],
                  paint: {
                    'line-color': '#ef4444',
                    'line-width': 3,
                    'line-opacity': 0.8,
                  },
                })
                
                // Apply visibility state
                if (!infrastructureVisible) {
                  map.current.setLayoutProperty(affectedLayerId, 'visibility', 'none')
                }
              }
              
              // Add popup handlers
              addPopupHandlers(layerId)
              if (map.current.getLayer(affectedLayerId)) {
                addPopupHandlers(affectedLayerId)
              }
            }
            
            // Ensure correct layer order after all layers are added:
            // hazard < infrastructure (green/unaffected) < affected (red)
            // Move infrastructure (green) above hazard first
            map.current.moveLayer(layerId)
            // If affected layer exists, move it to top (above infrastructure)
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
  }, [uploadedFile, analysisResult, mapLoaded, infrastructureVisible])

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
          <option value="esri-topo">Esri World Topo</option>
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

      {/* Map Container */}
      <div 
        ref={mapContainer} 
        className="w-full h-full" 
        style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
      />
    </div>
  )
}

