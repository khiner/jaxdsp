import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/FlameChart'
import ChartContext from './charts/ChartContext'

export default function Monitor({
  width = 400,
  height = 400,
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
            dimensions={{ x: 0, y: 200, width: 400, height: 200 }}
          />
        )}
        {traceTimeSeriesData && (
          <TimeSeriesChart data={traceTimeSeriesData} dimensions={{ x: 0, y: 0, width: 400, height: 200 }} />
        )}
      </ChartContext>
      {traceFlameData && <FlameChart data={traceFlameData} />}
    </div>
  )
}
