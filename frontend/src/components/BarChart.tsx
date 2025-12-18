import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { AnalysisResult } from '../types'

interface BarChartProps {
  data: AnalysisResult
}

// Custom tooltip component
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    const name = data.name
    const value = data.value
    const unit = data.unit || ''
    
    // Format currency for damage cost and remaining value
    const formatValue = (val: number) => {
      if (name === 'Damage Cost' || name === 'Remaining Value') {
        return val.toLocaleString('en-US', { 
          style: 'currency', 
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        })
      }
      return val.toLocaleString('en-US') + (unit ? ` ${unit}` : '')
    }
    
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-lg">
        <p className="text-white text-sm">{name}: {formatValue(value)}</p>
      </div>
    )
  }
  return null
}

export default function BarChart({ data }: BarChartProps) {
  const chartData = []

  // If vulnerability analysis is enabled, show damage cost and remaining value
  if (data.summary.total_damage_cost !== undefined && data.summary.total_damage_cost !== null) {
    const totalDamageCost = Math.max(0, data.summary.total_damage_cost || 0)
    const totalReplacementValue = Math.max(0, data.summary.total_replacement_value || 0)
    const remainingValue = Math.max(0, totalReplacementValue - totalDamageCost)
    
    chartData.push(
      {
        name: 'Damage Cost', // Full name for tooltip
        shortName: 'Damage', // Short name for X-axis
        value: totalDamageCost,
        fill: '#f59e0b',
        unit: ''
      },
      {
        name: 'Remaining Value', // Full name for tooltip
        shortName: 'Remaining', // Short name for X-axis
        value: remainingValue,
        fill: '#10b981',
        unit: ''
      }
    )
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
            dataKey={chartData.length === 2 && chartData[0].shortName ? "shortName" : "name"}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            label={{ fill: '#9ca3af' }}
            tickLine={false}
          />
          <YAxis 
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            label={{ fill: '#9ca3af' }}
            tickFormatter={(value) => {
              // Format as currency if showing damage cost or remaining value
              if (chartData.length === 2 && (chartData[0].name === 'Damage Cost' || chartData[1].name === 'Remaining Value')) {
                return value.toLocaleString('en-US', { 
                  style: 'currency', 
                  currency: 'USD',
                  minimumFractionDigits: 0,
                  maximumFractionDigits: 0,
                  notation: 'compact'
                })
              }
              return value.toLocaleString('en-US')
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" />
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  )
}

