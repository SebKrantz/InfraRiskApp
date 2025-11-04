import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { AnalysisResult } from '../types'

interface BarChartProps {
  data: AnalysisResult
}

// Custom tooltip component
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    const name = data.name.replace(' (m)', '') // Remove "(m)" suffix for tooltip
    const value = data.value
    
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-md px-3 py-2 shadow-lg">
        <p className="text-white text-sm">{name} : {value.toLocaleString()}</p>
      </div>
    )
  }
  return null
}

export default function BarChart({ data }: BarChartProps) {
  const chartData = []

  if (data.geometry_type === 'Point') {
    chartData.push(
      {
        name: 'Affected',
        value: data.summary.affected_count || 0,
        fill: '#ef4444'
      },
      {
        name: 'Unaffected',
        value: data.summary.unaffected_count || 0,
        fill: '#10b981'
      }
    )
  } else {
    chartData.push(
      {
        name: 'Affected (m)',
        value: data.summary.affected_meters || 0,
        fill: '#ef4444'
      },
      {
        name: 'Unaffected (m)',
        value: data.summary.unaffected_meters || 0,
        fill: '#10b981'
      }
    )
  }

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsBarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" />
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  )
}

