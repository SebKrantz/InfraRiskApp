import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ErrorBar } from 'recharts'
import { AnalysisResult } from '../types'

/** Recharts series name (tooltips); legend displays this as "Damage" via formatter */
const DAMAGE_COST_SERIES_NAME = 'Damage Cost (USD)'

const fmtUSD = (v: number) =>
  v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 })

interface BarChartProps {
  data: AnalysisResult
}

// Custom tooltip component
interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{ name: string; value: number; dataKey: string; payload?: { unit?: string } }>
  geometryType?: 'Point' | 'LineString'
  isExposureMode?: boolean
  damageLower?: number
  damageUpper?: number
}

const CustomTooltip = ({ active, payload, geometryType, isExposureMode, damageLower, damageUpper }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-lg">
        {payload.map((entry: any, index: number) => {
          const name = entry.name as string
          const tooltipLabel =
            name === DAMAGE_COST_SERIES_NAME
              ? 'Damage (USD)'
              : name === 'D. Ratio'
                ? 'Damage Ratio'
                : name
          const value = entry.value as number

          // Exposure bar (dataKey "value"): point = count only; line = meters
          if (isExposureMode && geometryType && entry.dataKey === 'value') {
            if (geometryType === 'Point') {
              return (
                <p key={index} className="text-white text-xs">
                  {value.toLocaleString('en-US')}
                </p>
              )
            }
            // Line: value is already in km
            return (
              <p key={index} className="text-white text-xs">
                {value.toLocaleString('en-US')} km
              </p>
            )
          }

          let formatted: string
          if (name === DAMAGE_COST_SERIES_NAME) {
            formatted = fmtUSD(value)
          } else if (name === 'Exposure' || name === 'Damage Ratio' || name === 'D. Ratio') {
            formatted = `${value.toFixed(1)}%`
          } else {
            formatted = value.toLocaleString('en-US')
          }

          return (
            <div key={index}>
              <p className="text-white text-xs">
                {tooltipLabel}: {formatted}
              </p>
              {name === DAMAGE_COST_SERIES_NAME && damageLower !== undefined && damageUpper !== undefined && (
                <p className="text-gray-400 text-xs">
                  Range: {fmtUSD(damageLower)} – {fmtUSD(damageUpper)}
                </p>
              )}
            </div>
          )
        })}
      </div>
    )
  }
  return null
}

export default function BarChart({ data }: BarChartProps) {
  const chartData: any[] = []
  const isVulnerabilityMode = data.summary.total_damage_cost !== undefined && data.summary.total_damage_cost !== null

  // Uncertainty bounds (populated only when a 4-column vulnerability CSV was used)
  const damageLower = data.summary.total_damage_cost_lower
  const damageUpper = data.summary.total_damage_cost_upper
  const hasErrorBars = damageLower !== undefined && damageUpper !== undefined

  // If vulnerability analysis is enabled, show damage cost (left axis) and exposure/vulnerability (right axis)
  if (isVulnerabilityMode) {
    const totalDamageCost = Math.max(0, data.summary.total_damage_cost || 0)

    // Calculate exposure percentage
    let exposurePct = 0
    if (data.geometry_type === 'Point') {
      const affectedCount = data.summary.affected_count || 0
      const totalFeatures = data.summary.total_features || 0
      exposurePct = totalFeatures > 0 ? (affectedCount / totalFeatures) * 100 : 0
    } else {
      const affectedMeters = data.summary.affected_meters || 0
      const unaffectedMeters = data.summary.unaffected_meters || 0
      const totalMeters = affectedMeters + unaffectedMeters
      exposurePct = totalMeters > 0 ? (affectedMeters / totalMeters) * 100 : 0
    }

    // Calculate average vulnerability of exposed assets (as percentage)
    let vulnerabilityPct = 0
    const infra = (data as any).infrastructure_features
    if (infra && infra.features && Array.isArray(infra.features)) {
      const features = infra.features as any[]

      if (data.geometry_type === 'Point') {
        // For points: simple average of vulnerability for affected points
        const exposed = features.filter(
          (f: any) => f.properties && f.properties.affected &&
          f.properties.vulnerability !== null && f.properties.vulnerability !== undefined
        )
        if (exposed.length > 0) {
          const sumV = exposed.reduce((acc: number, f: any) => acc + Number(f.properties.vulnerability || 0), 0)
          vulnerabilityPct = (sumV / exposed.length) * 100
        }
      } else {
        // For lines: length-weighted average vulnerability of exposed segments
        let weightedSum = 0
        let lengthSum = 0
        for (const f of features) {
          const props = f.properties || {}
          const affected = !!props.affected
          const vuln = props.vulnerability
          const lengthM = typeof props.length_m === 'number' ? props.length_m : 0
          if (affected && vuln !== null && vuln !== undefined && lengthM > 0) {
            weightedSum += Number(vuln) * lengthM
            lengthSum += lengthM
          }
        }
        if (lengthSum > 0) {
          vulnerabilityPct = (weightedSum / lengthSum) * 100
        }
      }
    }

    // Error bar values: Recharts ErrorBar expects [minus, plus] (distances from centre)
    const damageCostError = hasErrorBars
      ? [totalDamageCost - Math.max(0, damageLower!), Math.max(0, damageUpper!) - totalDamageCost]
      : undefined

    // Create a single data point with all three values
    chartData.push({
      name: '',
      damageCost: totalDamageCost,
      exposurePct,
      vulnerabilityPct,
      ...(damageCostError !== undefined ? { damageCostError } : {}),
    })
  } else {
    // Otherwise show affected/unaffected as before
    if (data.geometry_type === 'Point') {
      chartData.push(
        {
          name: 'Affected',
          value: data.summary.affected_count || 0,
          fill: '#ef4444',
          unit: ''
        },
        {
          name: 'Unaffected',
          value: data.summary.unaffected_count || 0,
          fill: '#10b981',
          unit: ''
        }
      )
    } else {
      chartData.push(
        {
          name: 'Affected',
          value: (data.summary.affected_meters || 0) / 1000,
          fill: '#ef4444',
          unit: 'km'
        },
        {
          name: 'Unaffected',
          value: (data.summary.unaffected_meters || 0) / 1000,
          fill: '#10b981',
          unit: 'km'
        }
      )
    }
  }

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsBarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#9ca3af" vertical={false} />
          <XAxis 
            dataKey="name"
            stroke="#9ca3af"
            tick={isVulnerabilityMode ? false : { fill: '#9ca3af' }}
            label={isVulnerabilityMode ? undefined : { fill: '#9ca3af' }}
            tickLine={false}
          />
          {/* Y axis: conditional based on mode */}
          {isVulnerabilityMode ? (
            <>
              {/* Left Y axis: damage cost (currency) */}
              <YAxis
                yAxisId="left"
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af' }}
                domain={hasErrorBars ? [0, (v: number) => Math.max(v, damageUpper! * 1.2)] : [0, 'auto']}
                tickFormatter={(value) => {
                  return value.toLocaleString('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0,
                    notation: 'compact'
                  })
                }}
              />
              {/* Right Y axis: percentages for exposure and vulnerability */}
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af' }}
                label={{ fill: '#9ca3af' }}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
                domain={[0, 100]}
              />
            </>
          ) : (
            <YAxis 
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af' }}
              label={{ fill: '#9ca3af' }}
              tickFormatter={(value) => value.toLocaleString('en-US')}
            />
          )}
          <Tooltip
            content={
              <CustomTooltip
                geometryType={data.geometry_type}
                isExposureMode={!isVulnerabilityMode}
                damageLower={damageLower}
                damageUpper={damageUpper}
              />
            }
          />
          {isVulnerabilityMode ? (
            <>
              <Bar
                yAxisId="left"
                dataKey="damageCost"
                name={DAMAGE_COST_SERIES_NAME}
                fill="#f59e0b"
              >
                {hasErrorBars && (
                  <ErrorBar
                    dataKey="damageCostError"
                    width={5}
                    strokeWidth={2}
                    stroke="#374151"
                    direction="y"
                  />
                )}
              </Bar>
              <Bar
                yAxisId="right"
                dataKey="exposurePct"
                name="Exposure"
                fill="#3b82f6"
              />
              <Bar
                yAxisId="right"
                dataKey="vulnerabilityPct"
                name="D. Ratio"
                fill="#10b981"
              />
              <Legend
                verticalAlign="bottom"
                align="center"
                wrapperStyle={{
                  paddingLeft: 0,
                  paddingRight: 0,
                  whiteSpace: 'nowrap',
                }}
                formatter={(value: string) =>
                  value === DAMAGE_COST_SERIES_NAME ? 'Damage' : value
                }
              />
            </>
          ) : (
            <Bar dataKey="value" />
          )}
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  )
}

