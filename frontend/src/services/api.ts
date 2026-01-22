import { Hazard, UploadedFile, AnalysisResult } from '../types'

const API_BASE_URL = 'api'

/**
 * Upload a spatial file
 */
export async function uploadFile(file: File): Promise<UploadedFile> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || 'Failed to upload file')
  }

  return response.json()
}

/**
 * Get list of available hazard layers
 */
export async function getHazards(): Promise<Hazard[]> {
  const response = await fetch(`${API_BASE_URL}/hazards`)

  if (!response.ok) {
    throw new Error('Failed to fetch hazard layers')
  }

  const data = await response.json()
  return data.hazards || []
}

/**
 * Get specific hazard layer info
 */
export async function getHazardInfo(hazardId: string): Promise<Hazard> {
  const response = await fetch(`${API_BASE_URL}/hazards/${hazardId}`)

  if (!response.ok) {
    throw new Error('Failed to fetch hazard layer info')
  }

  return response.json()
}

/**
 * Analyze intersections between infrastructure and hazard
 */
export async function analyze(
  fileId: string,
  hazardId: string,
  hazardUrl: string,
  intensityThreshold?: number,
  vulnerabilityCurveFile?: File | null,
  replacementValue?: number | null
): Promise<AnalysisResult> {
  // Use FormData if we have a vulnerability curve file, otherwise use JSON
  const hasVulnerabilityData = vulnerabilityCurveFile !== null && vulnerabilityCurveFile !== undefined
  
  let body: FormData | string
  let headers: HeadersInit
  
  if (hasVulnerabilityData) {
    const formData = new FormData()
    formData.append('file_id', fileId)
    formData.append('hazard_id', hazardId)
    formData.append('hazard_url', hazardUrl)
    if (intensityThreshold !== undefined) {
      formData.append('intensity_threshold', intensityThreshold.toString())
    }
    formData.append('vulnerability_curve_file', vulnerabilityCurveFile)
    if (replacementValue !== null && replacementValue !== undefined) {
      formData.append('replacement_value', replacementValue.toString())
    }
    body = formData
    headers = {} // Let browser set Content-Type with boundary for FormData
  } else {
    body = JSON.stringify({
      file_id: fileId,
      hazard_id: hazardId,
      hazard_url: hazardUrl,
      intensity_threshold: intensityThreshold,
    })
    headers = {
      'Content-Type': 'application/json',
    }
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers,
    body,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Analysis failed' }))
    throw new Error(error.detail || 'Failed to perform analysis')
  }

  return response.json()
}

/**
 * Get upload info
 */
export async function getUploadInfo(fileId: string): Promise<UploadedFile> {
  const response = await fetch(`${API_BASE_URL}/upload/${fileId}`)

  if (!response.ok) {
    throw new Error('Failed to fetch upload info')
  }

  return response.json()
}

/**
 * Get hazard raster statistics (min, max)
 */
export async function getHazardStats(hazardId: string): Promise<{ min: number; max: number }> {
  const response = await fetch(`${API_BASE_URL}/hazards/${hazardId}/stats`)

  if (!response.ok) {
    throw new Error('Failed to fetch hazard statistics')
  }

  return response.json()
}

/**
 * Export analysis results as a high-resolution PNG barchart
 */
export async function exportBarchart(
  fileId: string,
  hazardId: string,
  intensityThreshold?: number
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/export/barchart`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id: fileId,
      hazard_id: hazardId,
      intensity_threshold: intensityThreshold,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Export failed' }))
    throw new Error(error.detail || 'Failed to export barchart')
  }

  // Get filename from Content-Disposition header or generate one
  const contentDisposition = response.headers.get('Content-Disposition')
  let filename = `barchart_${fileId.slice(0, 8)}_${hazardId.slice(0, 8)}.png`
  if (contentDisposition) {
    const filenameMatch = contentDisposition.match(/filename="(.+)"/)
    if (filenameMatch) {
      filename = filenameMatch[1]
    }
  }

  // Download the blob
  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

/**
 * Export analysis results as a high-resolution PNG map
 */
export async function exportMap(
  fileId: string,
  hazardId: string,
  colorPalette: string,
  hazardOpacity: number,
  intensityThreshold?: number
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/export/map`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id: fileId,
      hazard_id: hazardId,
      color_palette: colorPalette,
      hazard_opacity: hazardOpacity / 100, // Convert percentage to 0-1
      intensity_threshold: intensityThreshold,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Export failed' }))
    throw new Error(error.detail || 'Failed to export map')
  }

  // Get filename from Content-Disposition header or generate one
  const contentDisposition = response.headers.get('Content-Disposition')
  let filename = `map_${fileId.slice(0, 8)}_${hazardId.slice(0, 8)}.png`
  if (contentDisposition) {
    const filenameMatch = contentDisposition.match(/filename="(.+)"/)
    if (filenameMatch) {
      filename = filenameMatch[1]
    }
  }

  // Download the blob
  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

