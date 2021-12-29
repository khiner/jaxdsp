import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/FlameChart'
import ChartContext from './charts/ChartContext'

export default function Monitor({
  width = 400,
  height = 500,
  trainTimeSeriesData,
  traceTimeSeriesData,
  traceFlameData,
}) {
  if (!trainTimeSeriesData && !traceTimeSeriesData && !traceFlameData) return null

  return (
    <div>
      <ChartContext width={width} height={height}>
        {trainTimeSeriesData && (
          <TimeSeriesChart
            data={trainTimeSeriesData}
            dimensions={{ x: 0, y: 300, width: 400, height: 200 }}
          />
        )}
        {traceTimeSeriesData && (
          <TimeSeriesChart
            data={traceTimeSeriesData}
            dimensions={{ x: 0, y: 100, width: 400, height: 200 }}
          />
        )}
        {traceFlameData && (
          <FlameChart data={traceFlameData} dimensions={{ x: 0, y: 0, width: 400, height: 100 }} />
        )}
      </ChartContext>
    </div>
  )
}
