import { useState } from 'react'
import { ChevronLeft, ChevronRight, Info, ChevronUp, ChevronDown, Download } from 'lucide-react'
import { Hazard, UploadedFile, AnalysisResult, ColorPalette } from '../types'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Select } from './ui/select'
import { Slider } from './ui/slider'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog'
import BarChart from './BarChart'
import { exportBarchart, exportMap } from '../services/api'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  uploadedFile: UploadedFile | null
  onFileUpload: (file: File) => Promise<void>
  onClearData: () => void
  hazards: Hazard[]
  selectedHazard: Hazard | null
  onSelectHazard: (hazard: Hazard | null) => void
  colorPalette: ColorPalette
  onColorPaletteChange: (palette: ColorPalette) => void
  hazardOpacity: number
  onHazardOpacityChange: (opacity: number) => void
  intensityThreshold: number
  onIntensityThresholdChange: (threshold: number) => void
  hazardStats: { min: number; max: number } | null
  analysisResult: AnalysisResult | null
  loadingUpload?: boolean
  loadingAnalysis?: boolean
  error?: string | null
}

const colorPalettes: ColorPalette[] = ['viridis', 'magma', 'inferno', 'plasma', 'cividis', 'turbo']

export default function Sidebar({
  isOpen,
  onToggle,
  uploadedFile,
  onFileUpload,
  onClearData,
  hazards,
  selectedHazard,
  onSelectHazard,
  colorPalette,
  onColorPaletteChange,
  hazardOpacity,
  onHazardOpacityChange,
  intensityThreshold,
  onIntensityThresholdChange,
  hazardStats,
  analysisResult,
  loadingUpload = false,
  loadingAnalysis = false,
  error = null,
}: SidebarProps) {
  const [infoOpen, setInfoOpen] = useState(false)
  const [uploadInfoOpen, setUploadInfoOpen] = useState(false)
  const [thresholdInfoOpen, setThresholdInfoOpen] = useState(false)
  const [paletteInfoOpen, setPaletteInfoOpen] = useState(false)
  const [exportingBarchart, setExportingBarchart] = useState(false)
  const [exportingMap, setExportingMap] = useState(false)

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    await onFileUpload(file)
    // Reset file input
    e.target.value = ''
  }

  const handleHazardSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const hazardId = e.target.value
    const hazard = hazards.find(h => h.id === hazardId) || null
    onSelectHazard(hazard)
  }

  return (
    <>
      <div
        className={`bg-gray-900 transition-all duration-300 flex flex-col ${
          isOpen ? 'w-80' : 'w-0 overflow-hidden'
        }`}
      >
        {isOpen && (
          <div className="flex flex-col h-full overflow-y-auto p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
              <h1 className="text-xl font-bold text-white">Infrastructure Risk Analyzer</h1>
              <Button variant="ghost" size="sm" onClick={onToggle} className="text-gray-300 hover:text-white hover:bg-gray-800">
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </div>

            {/* Input Fields Section */}
            <div className="flex flex-col flex-1">
              {/* File Upload */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm font-medium text-gray-300">
                    Upload Infrastructure Data
                  </label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setUploadInfoOpen(true)}
                    className="h-6 w-6 p-0 text-gray-300 hover:text-white hover:bg-gray-800"
                  >
                    <Info className="h-3.5 w-3.5" />
                  </Button>
                </div>
                {uploadedFile ? (
                  <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-gray-300 truncate">
                          ✓ {uploadedFile.filename}
                        </p>
                        <p className="text-xs text-gray-400 mt-0.5">
                          {uploadedFile.feature_count} features
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={onClearData}
                        disabled={loadingUpload || loadingAnalysis}
                        className="ml-2 text-xs text-gray-300 bg-gray-700 hover:text-red-500 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Clear Data
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-sm">
                      <Input
                        type="file"
                        accept=".shp,.gpkg,.csv,.zip"
                        onChange={handleFileChange}
                        disabled={loadingUpload}
                        className="cursor-pointer border-0 bg-transparent px-0 py-0 h-auto text-gray-300 file:mr-4 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-600 file:text-white hover:file:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed [&::file-selector-button]:mr-4"
                      />
                    </div>
                    {loadingUpload && (
                      <p className="text-xs text-blue-400 mt-1">Uploading...</p>
                    )}
                    {error && (
                      <p className="text-xs text-red-400 mt-1">{error}</p>
                    )}
                  </>
                )}
              </div>

              {/* Hazard Selector */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm font-medium text-gray-300">
                    Select Hazard Layer
                  </label>
                  {selectedHazard && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setInfoOpen(true)}
                      className="h-6 w-6 p-0 text-gray-300 hover:text-white hover:bg-gray-800"
                    >
                      <Info className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-sm">
                  <Select value={selectedHazard?.id || ''} onChange={handleHazardSelect} className="border-0 bg-transparent px-0 py-0 h-auto text-gray-300">
                    <option value="" className="bg-gray-800 text-gray-300">Select a hazard...</option>
                    {hazards.map(hazard => (
                      <option key={hazard.id} value={hazard.id} className="bg-gray-800 text-gray-300">
                        {hazard.name}
                      </option>
                    ))}
                  </Select>
                </div>
              </div>

              {/* Color Palette and Opacity */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm font-medium text-gray-300">
                    Color Palette for Hazard Layer
                  </label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setPaletteInfoOpen(true)}
                    className="h-6 w-6 p-0 text-gray-300 hover:text-white hover:bg-gray-800"
                  >
                    <Info className="h-3.5 w-3.5" />
                  </Button>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-sm flex items-center gap-3">
                  <Select
                    value={colorPalette}
                    onChange={(e) => onColorPaletteChange(e.target.value as ColorPalette)}
                    className="border-0 bg-transparent px-0 py-0 h-auto text-gray-300 flex-1"
                  >
                    {colorPalettes.map(palette => (
                      <option key={palette} value={palette} className="bg-gray-800 text-gray-300">
                        {palette.charAt(0).toUpperCase() + palette.slice(1)}
                      </option>
                    ))}
                  </Select>
                  <div className="flex items-center border-l border-gray-700 pl-3">
                    <div className="relative flex items-center">
                      <Input
                        type="number"
                        min="0"
                        max="100"
                        value={hazardOpacity}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 0
                          if (val >= 0 && val <= 100) {
                            onHazardOpacityChange(val)
                          }
                        }}
                        className="opacity-input w-16 border-0 bg-transparent px-2 py-0 h-auto text-gray-300 text-sm text-right"
                      />
                      <span className="text-xs text-gray-400">%</span>
                      <div className="flex flex-col ml-1 gap-0">
                        <button
                          type="button"
                          onClick={() => {
                            if (hazardOpacity < 100) {
                              onHazardOpacityChange(hazardOpacity + 1)
                            }
                          }}
                          disabled={hazardOpacity >= 100}
                          className="p-0 text-gray-400 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                          <ChevronUp className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (hazardOpacity > 0) {
                              onHazardOpacityChange(hazardOpacity - 1)
                            }
                          }}
                          disabled={hazardOpacity <= 0}
                          className="p-0 text-gray-400 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                          <ChevronDown className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Intensity Slider */}
              {selectedHazard && hazardStats && (
                <div className="mb-3">
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-sm font-medium text-gray-300">
                      Hazard Intensity Threshold
                    </label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setThresholdInfoOpen(true)}
                      className="h-6 w-6 p-0 text-gray-300 hover:text-white hover:bg-gray-800"
                    >
                      <Info className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <div className="space-y-1">
                    <Slider
                      value={intensityThreshold}
                      onValueChange={onIntensityThresholdChange}
                      min={hazardStats.min}
                      max={hazardStats.max}
                      step={(hazardStats.max - hazardStats.min) / 1000}
                    />
                    <div className="flex justify-between text-xs text-gray-400">
                      <span>{hazardStats.min.toFixed(2)}</span>
                      <span>Current: {intensityThreshold.toFixed(2)}</span>
                      <span>{hazardStats.max.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Horizontal Separator */}
            <hr className="my-4 border-gray-700" />

            {/* Bar Chart Section */}
            <div className="flex-shrink-0">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Analysis Results</h3>
              {loadingAnalysis && !analysisResult && (
                <div className="w-full h-48 flex items-center justify-center text-sm text-gray-500 border border-gray-700 rounded bg-gray-800">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                    <p>Analyzing...</p>
                  </div>
                </div>
              )}
              {!loadingAnalysis && analysisResult ? (
                <BarChart data={analysisResult} />
              ) : !loadingAnalysis && !analysisResult ? (
                <div className="w-full h-48 flex items-center justify-center text-sm text-gray-500 border border-gray-700 rounded bg-gray-800">
                  {uploadedFile && selectedHazard ? 'No analysis data yet' : 'Upload data and select a hazard to analyze'}
                </div>
              ) : null}
              {error && (
                <p className="text-xs text-red-400 mt-2">{error}</p>
              )}
              
              {/* Export Buttons */}
              {!loadingAnalysis && analysisResult && uploadedFile && selectedHazard && (
                <div className="mt-4 flex gap-2">
                  <Button
                    onClick={async () => {
                      if (!uploadedFile || !selectedHazard) return
                      try {
                        setExportingBarchart(true)
                        const threshold = hazardStats && intensityThreshold > hazardStats.min 
                          ? intensityThreshold 
                          : undefined
                        await exportBarchart(
                          uploadedFile.file_id,
                          selectedHazard.id,
                          threshold
                        )
                      } catch (err) {
                        console.error('Failed to export barchart:', err)
                        alert(err instanceof Error ? err.message : 'Failed to export barchart')
                      } finally {
                        setExportingBarchart(false)
                      }
                    }}
                    disabled={exportingBarchart || exportingMap}
                    className="flex-1 bg-gray-800 border border-gray-700 hover:bg-gray-700 text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {exportingBarchart ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-300 mr-2"></div>
                        Generating...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        Barchart
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={async () => {
                      if (!uploadedFile || !selectedHazard) return
                      try {
                        setExportingMap(true)
                        const threshold = hazardStats && intensityThreshold > hazardStats.min 
                          ? intensityThreshold 
                          : undefined
                        await exportMap(
                          uploadedFile.file_id,
                          selectedHazard.id,
                          colorPalette,
                          hazardOpacity,
                          threshold
                        )
                      } catch (err) {
                        console.error('Failed to export map:', err)
                        alert(err instanceof Error ? err.message : 'Failed to export map')
                      } finally {
                        setExportingMap(false)
                      }
                    }}
                    disabled={exportingBarchart || exportingMap}
                    className="flex-1 bg-gray-800 border border-gray-700 hover:bg-gray-700 text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {exportingMap ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-300 mr-2"></div>
                        Generating...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        Map
                      </>
                    )}
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Collapse/Expand Button */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-gray-900 rounded-r-lg p-2 shadow-lg hover:bg-gray-800 text-white"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      )}

      {/* Hazard Info Dialog */}
      <Dialog open={infoOpen} onOpenChange={setInfoOpen}>
        <DialogContent onClose={() => setInfoOpen(false)}>
          <DialogHeader>
            <DialogTitle>{selectedHazard?.name}</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            {selectedHazard?.description && (
              <p className="text-sm text-gray-700">{selectedHazard.description}</p>
            )}
            {selectedHazard?.metadata && (
              <div style={{ paddingTop: '0.7rem' }}>
                <p className="text-sm font-bold text-gray-600 mb-1">Background Paper</p>
                <p className="text-sm text-gray-700">
                  {(() => {
                    const text = selectedHazard.metadata
                    // URL regex pattern
                    const urlRegex = /(https?:\/\/[^\s]+)/g
                    const parts = text.split(urlRegex)
                    
                    return parts.map((part, index) => {
                      // Check if part matches URL pattern
                      const isUrl = /^https?:\/\/[^\s]+$/.test(part)
                      if (isUrl) {
                        return (
                          <a
                            key={index}
                            href={part}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline"
                          >
                            {part}
                          </a>
                        )
                      }
                      return <span key={index}>{part}</span>
                    })
                  })()}
                </p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Upload Info Dialog */}
      <Dialog open={uploadInfoOpen} onOpenChange={setUploadInfoOpen}>
        <DialogContent onClose={() => setUploadInfoOpen(false)}>
          <DialogHeader>
            <DialogTitle>Accepted File Formats</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">Shapefile (.shp)</p>
              <p className="text-sm text-gray-600">
                Upload as a ZIP file containing all required components (.shp, .shx, .dbf, and optionally .prj).
              </p>
            </div>
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">GeoPackage (.gpkg)</p>
              <p className="text-sm text-gray-600">
                Single-file geospatial database format. Supports both Point and LineString geometries.
              </p>
            </div>
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">CSV (.csv)</p>
              <p className="text-sm text-gray-600">
                Comma or semicolon-delimited file with coordinate columns. Supported column names:
              </p>
              <ul className="text-xs text-gray-500 mt-1 ml-4 list-disc">
                <li>lat/lon, lat/lng</li>
                <li>latitude/longitude</li>
                <li>y/x (case-insensitive)</li>
              </ul>
              <p className="text-xs text-gray-500 mt-1">
                CSV files will be interpreted as Point geometries.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-300">
              <p className="text-xs text-gray-500">
                <strong>Note:</strong> Maximum file size is 100 MB. Supported geometry types are Point and LineString.
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Threshold Info Dialog */}
      <Dialog open={thresholdInfoOpen} onOpenChange={setThresholdInfoOpen}>
        <DialogContent onClose={() => setThresholdInfoOpen(false)}>
          <DialogHeader>
            <DialogTitle>Hazard Intensity Threshold</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-700">
                The intensity threshold determines the minimum hazard value required for infrastructure to be considered <strong>"affected"</strong>.
              </p>
            </div>
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">How it works:</p>
              <ul className="text-sm text-gray-600 space-y-1 ml-4 list-disc">
                <li>
                  <strong>Point features:</strong> A point is marked as affected (red) if the hazard raster value at that location is greater than or equal to the threshold.
                </li>
                <li>
                  <strong>Line features:</strong> A line is marked as affected (red) if any point along the line has a hazard value greater than or equal to the threshold.
                </li>
              </ul>
            </div>
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">Impact on results:</p>
              <ul className="text-sm text-gray-600 space-y-1 ml-4 list-disc">
                <li>
                  <strong>Bar chart:</strong> Shows the count (for points) or length in meters (for lines) of affected vs. unaffected infrastructure based on the threshold.
                </li>
                <li>
                  <strong>Map:</strong> Infrastructure features are color-coded: <span className="text-red-600 font-semibold">red</span> for affected and <span className="text-green-600 font-semibold">green</span> for unaffected.
                </li>
              </ul>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-300">
              <p className="text-xs text-gray-500">
                <strong>Tip:</strong> Adjust the slider to explore different risk scenarios. A lower threshold means more infrastructure will be considered at risk.
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Palette Info Dialog */}
      <Dialog open={paletteInfoOpen} onOpenChange={setPaletteInfoOpen}>
        <DialogContent onClose={() => setPaletteInfoOpen(false)}>
          <DialogHeader>
            <DialogTitle>Color Palette & Opacity</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">Color Palette:</p>
              <p className="text-sm text-gray-600">
                The color palette determines how hazard intensity values are visualized on the map. Different palettes are optimized for different types of data:
              </p>
              <ul className="text-sm text-gray-600 space-y-1 mt-2 ml-4 list-disc">
                <li><strong>Viridis:</strong> Perceptually uniform, colorblind-friendly (blue to yellow)</li>
                <li><strong>Magma:</strong> Black to purple to pink to yellow</li>
                <li><strong>Inferno:</strong> Black to red to yellow</li>
                <li><strong>Plasma:</strong> Purple to pink to yellow</li>
                <li><strong>Cividis:</strong> Dark to bright, with good contrast</li>
                <li><strong>Turbo:</strong> Rainbow-like spectrum (blue to yellow via red)</li>
              </ul>
              {/* <p className="text-sm text-gray-600 mt-2">
                Colors are automatically scaled from the minimum to maximum hazard values in your dataset. Lower values map to the left side of the palette, higher values to the right.
              </p> */}
            </div>
            <div>
              <p className="text-sm font-bold text-gray-700 mb-1">Opacity:</p>
              <p className="text-sm text-gray-600">
                The opacity percentage controls the transparency of the hazard layer. 
              </p>
              {/* <ul className="text-sm text-gray-600 space-y-1 mt-2 ml-4 list-disc">
                <li>View the underlying basemap through the hazard layer</li>
                <li>Compare hazard intensity with infrastructure features</li>
                <li>Adjust visibility for better visual clarity</li>
              </ul> */}
              <p className="text-sm text-gray-600 mt-2">
                <strong>0%</strong> = fully transparent (hazard layer invisible)<br />
                <strong>100%</strong> = fully opaque (hazard layer completely covers basemap)
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-300">
              <p className="text-xs text-gray-500">
                <strong>Tip:</strong> Experiment with different palettes and opacity levels to find the best visualization for your analysis. The hazard layer is displayed between the basemap and infrastructure layers.
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

