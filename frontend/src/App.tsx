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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch hazards on mount
  useEffect(() => {
    const fetchHazards = async () => {
      try {
        setLoading(true)
        const hazardsData = await getHazards()
        setHazards(hazardsData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load hazards')
      } finally {
        setLoading(false)
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
        setLoading(true)
        const stats = await getHazardStats(selectedHazard.id)
        setHazardStats({ min: stats.min, max: stats.max })
        // Set threshold to min value initially
        setIntensityThreshold(stats.min)
      } catch (err) {
        console.error('Failed to fetch hazard statistics:', err)
        // Fallback to default range if stats fail
        setHazardStats({ min: 0, max: 100 })
      } finally {
        setLoading(false)
      }
    }
    fetchStats()
  }, [selectedHazard])

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true)
      setError(null)
      const uploaded = await uploadFile(file)
      setUploadedFile(uploaded)
      // Clear previous analysis when new file is uploaded
      setAnalysisResult(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file')
      setUploadedFile(null)
    } finally {
      setLoading(false)
    }
  }

  // Handle clear data
  const handleClearData = () => {
    setUploadedFile(null)
    setAnalysisResult(null)
    setError(null)
  }

  // Perform analysis when hazard or threshold changes
  useEffect(() => {
    if (!uploadedFile || !selectedHazard) {
      setAnalysisResult(null)
      return
    }

    const performAnalysis = async () => {
      try {
        setLoading(true)
        setError(null)
        // Use threshold only if it's greater than the min value
        const threshold = hazardStats && intensityThreshold > hazardStats.min 
          ? intensityThreshold 
          : undefined
        
        const result = await analyzeApi(
          uploadedFile.file_id,
          selectedHazard.id,
          selectedHazard.url,
          threshold
        )
        setAnalysisResult(result)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to perform analysis')
        setAnalysisResult(null)
      } finally {
        setLoading(false)
      }
    }

    // Debounce analysis calls
    const timeoutId = setTimeout(performAnalysis, 300)
    return () => clearTimeout(timeoutId)
  }, [uploadedFile, selectedHazard, intensityThreshold])

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
        loading={loading}
        error={error}
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
        loading={loading}
      />
    </div>
  )
}

export default App

