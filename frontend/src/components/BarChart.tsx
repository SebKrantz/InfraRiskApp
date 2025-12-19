import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { AnalysisResult } from '../types'

interface BarChartProps {
  data: AnalysisResult
}

// Custom tooltip component
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-lg">
        {payload.map((entry: any, index: number) => {
          const name = entry.name
          const value = entry.value as number
          
          let formatted: string
          if (name === 'Damage Cost') {
            formatted = value.toLocaleString('en-US', { 
              style: 'currency', 
              currency: 'USD',
              minimumFractionDigits: 0,
              maximumFractionDigits: 0
            })
          } else if (name === 'Exposure' || name === 'Vulnerability') {
            formatted = `${value.toFixed(1)}%`
          } else {
            formatted = value.toLocaleString('en-US')
          }
          
          return (
            <p key={index} className="text-white text-xs">
              {name}: {formatted}
            </p>
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

    // Create a single data point with all three values
    chartData.push({
      name: '',
      damageCost: totalDamageCost,
      exposurePct,
      vulnerabilityPct
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
                label={{ fill: '#9ca3af' }}
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
          <Tooltip content={<CustomTooltip />} />
          {isVulnerabilityMode ? (
            <>
              <Bar
                yAxisId="left"
                dataKey="damageCost"
                name="Damage Cost"
                fill="#f59e0b"
              />
              <Bar
                yAxisId="right"
                dataKey="exposurePct"
                name="Exposure"
                fill="#3b82f6"
              />
              <Bar
                yAxisId="right"
                dataKey="vulnerabilityPct"
                name="Vulnerability"
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
                  value === 'Damage Cost' ? 'Damage' : value
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

