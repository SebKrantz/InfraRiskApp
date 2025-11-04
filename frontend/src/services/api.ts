import { Hazard, UploadedFile, AnalysisResult } from '../types'

const API_BASE_URL = '/api'

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
  intensityThreshold?: number
): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id: fileId,
      hazard_id: hazardId,
      hazard_url: hazardUrl,
      intensity_threshold: intensityThreshold,
    }),
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

