import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import MapView from './components/MapView'
import { Hazard, UploadedFile, AnalysisResult, ColorPalette, Basemap } from './types'
import { getHazards, uploadFile, analyze as analyzeApi, getHazardStats } from './services/api'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [selectedHazard, setSelectedHazard] = useState<Hazard | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [colorPalette, setColorPalette] = useState<ColorPalette>('turbo')
  const [intensityThreshold, setIntensityThreshold] = useState<number>(0)
  const [hazardOpacity, setHazardOpacity] = useState<number>(60)
  const [hazardStats, setHazardStats] = useState<{ min: number; max: number } | null>(null)
  const [basemap, setBasemap] = useState<Basemap>('positron')
  const [hazards, setHazards] = useState<Hazard[]>([])
  const [loadingUpload, setLoadingUpload] = useState(false)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [vulnerabilityAnalysisEnabled, setVulnerabilityAnalysisEnabled] = useState(false)
  const [vulnerabilityCurveFile, setVulnerabilityCurveFile] = useState<File | null>(null)
  const [replacementValue, setReplacementValue] = useState<number | null>(null)

  // Fetch hazards on mount
  useEffect(() => {
    const fetchHazards = async () => {
      try {
        const hazardsData = await getHazards()
        setHazards(hazardsData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load hazards')
      }
    }
    fetchHazards()
  }, [])

  // Fetch hazard statistics when hazard is selected
  useEffect(() => {
    const fetchStats = async () => {
      if (!selectedHazard) {
        setHazardStats(null)
        setIntensityThreshold(0)
        return
      }

      try {
        const stats = await getHazardStats(selectedHazard.id)
        setHazardStats({ min: stats.min, max: stats.max })
        // Set threshold to min value initially
        setIntensityThreshold(stats.min)
      } catch (err) {
        console.error('Failed to fetch hazard statistics:', err)
        // Fallback to default range if stats fail
        setHazardStats({ min: 0, max: 100 })
      }
    }
    fetchStats()
  }, [selectedHazard])

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    try {
      setLoadingUpload(true)
      setError(null)
      const uploaded = await uploadFile(file)
      setUploadedFile(uploaded)
      // Clear previous analysis when new file is uploaded
      setAnalysisResult(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file')
      setUploadedFile(null)
    } finally {
      setLoadingUpload(false)
    }
  }

  // Handle clear data
  const handleClearData = () => {
    setUploadedFile(null)
    setAnalysisResult(null)
    setError(null)
  }

  // Perform analysis when hazard, threshold, or vulnerability analysis changes
  useEffect(() => {
    if (!uploadedFile || !selectedHazard) {
      setAnalysisResult(null)
      return
    }

    // If vulnerability analysis is enabled, require both curve file and replacement value
    if (vulnerabilityAnalysisEnabled && (!vulnerabilityCurveFile || replacementValue === null)) {
      // Don't trigger analysis if vulnerability analysis is enabled but missing required fields
      return
    }

    const performAnalysis = async () => {
      try {
        setLoadingAnalysis(true)
        setError(null)
        // Use threshold only if it's greater than the min value
        const threshold = hazardStats && intensityThreshold > hazardStats.min 
          ? intensityThreshold 
          : undefined
        
        const result = await analyzeApi(
          uploadedFile.file_id,
          selectedHazard.id,
          selectedHazard.url,
          threshold,
          vulnerabilityAnalysisEnabled ? vulnerabilityCurveFile : null,
          vulnerabilityAnalysisEnabled ? replacementValue : null
        )
        setAnalysisResult(result)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to perform analysis')
        setAnalysisResult(null)
      } finally {
        setLoadingAnalysis(false)
      }
    }

    // Debounce analysis calls
    const timeoutId = setTimeout(performAnalysis, 300)
    return () => clearTimeout(timeoutId)
  }, [uploadedFile, selectedHazard, intensityThreshold, vulnerabilityAnalysisEnabled, vulnerabilityCurveFile, replacementValue])

  return (
    <div className="flex h-screen w-screen overflow-hidden m-0 p-0">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        uploadedFile={uploadedFile}
        onFileUpload={handleFileUpload}
        onClearData={handleClearData}
        hazards={hazards}
        selectedHazard={selectedHazard}
        onSelectHazard={setSelectedHazard}
        colorPalette={colorPalette}
        onColorPaletteChange={setColorPalette}
        hazardOpacity={hazardOpacity}
        onHazardOpacityChange={setHazardOpacity}
        intensityThreshold={intensityThreshold}
        onIntensityThresholdChange={setIntensityThreshold}
        hazardStats={hazardStats}
        analysisResult={analysisResult}
        loadingUpload={loadingUpload}
        loadingAnalysis={loadingAnalysis}
        error={error}
        vulnerabilityAnalysisEnabled={vulnerabilityAnalysisEnabled}
        onVulnerabilityAnalysisEnabledChange={setVulnerabilityAnalysisEnabled}
        vulnerabilityCurveFile={vulnerabilityCurveFile}
        onVulnerabilityCurveFileChange={setVulnerabilityCurveFile}
        replacementValue={replacementValue}
        onReplacementValueChange={setReplacementValue}
      />
      <MapView
        uploadedFile={uploadedFile}
        selectedHazard={selectedHazard}
        analysisResult={analysisResult}
        colorPalette={colorPalette}
        intensityThreshold={intensityThreshold}
        hazardOpacity={hazardOpacity}
        basemap={basemap}
        onBasemapChange={setBasemap}
        loadingAnalysis={loadingAnalysis}
      />
    </div>
  )
}

export default App

