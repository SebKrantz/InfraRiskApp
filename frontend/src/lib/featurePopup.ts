// The map's feature pop-up: the HTML MapLibre's Popup.setHTML renders. Column names and
// values come from the user's file, so every one of them is escaped — a column named
// <img src=x onerror=...> would otherwise run as script on the app's origin. Numeric
// fields (length, damage) go through Number() for the same reason.

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

export function formatFeatureProperties(
  properties: Record<string, any>,
  vulnerabilityAnalysisEnabled: boolean,
  unit?: string
): string {
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
      // Truncate very long strings, then escape
      return escapeHtml(value.length > 100 ? value.substring(0, 100) + '...' : value)
    }
    return escapeHtml(String(value))
  }

  const formatExposureLevel = (value: any, unit?: string): string => {
    if (value === null || value === undefined) {
      return '<span class="text-gray-400 italic">N/A</span>'
    }
    if (typeof value === 'number') {
      // Decimals by magnitude, as on the hazard legend: none from 100, one from 1, two below
      const abs = Math.abs(value)
      const base = value.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: abs >= 100 ? 0 : abs >= 1 ? 1 : 2
      })
      return unit ? `${base} ${unit}` : base
    }
    return escapeHtml(String(value))
  }

  const formatPropertyName = (key: string): string => {
    // Convert snake_case and camelCase to Title Case
    return escapeHtml(key
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, (str) => str.toUpperCase())
      .trim())
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
  const damageCostLower = properties['damage_cost_lower']
  const damageCostUpper = properties['damage_cost_upper']

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
    computedFeatures.push({ label: 'Length:', value: formatLength(Number(lengthM)) })
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
      computedFeatures.push({ label: 'Max Exposure Level:', value: formatExposureLevel(exposureLevelMax, unit) })
    }
    if (exposureLevelAvg !== undefined && exposureLevelAvg !== null) {
      computedFeatures.push({ label: 'Avg Exposure Level:', value: formatExposureLevel(exposureLevelAvg, unit) })
    }
  } else if (exposureLevel !== undefined && exposureLevel !== null) {
    // Point feature - show single exposure level
    computedFeatures.push({ label: 'Exposure Level:', value: formatExposureLevel(exposureLevel, unit) })
  }

  // Add vulnerability and damage cost if available
  if (vulnerability !== undefined && vulnerability !== null) {
    const formatVulnerability = (value: number): string => {
      // Format as percentage with 1-2 decimal places
      const percentage = value * 100
      return `${percentage.toFixed(percentage < 1 ? 2 : 1)}%`
    }
    computedFeatures.push({ label: 'Damage Ratio:', value: formatVulnerability(Number(vulnerability)) })
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
    computedFeatures.push({ label: 'Damage Cost (USD):', value: formatDamageCost(Number(damageCost)) })
    if (damageCostLower !== undefined && damageCostLower !== null &&
        damageCostUpper !== undefined && damageCostUpper !== null) {
      computedFeatures.push({
        label: 'Range:',
        value: `${formatDamageCost(Number(damageCostLower))} – ${formatDamageCost(Number(damageCostUpper))}`
      })
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
  // Exclude affected, exposure level, vulnerability, and damage_cost properties from the main list since they're shown at the top
  const sortedKeys = Object.keys(properties)
    .filter(key => !['affected', 'exposure_level', 'exposure_level_max', 'exposure_level_avg', 'length_m', 'vulnerability', 'damage_cost', 'damage_cost_lower', 'damage_cost_upper'].includes(key))
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
