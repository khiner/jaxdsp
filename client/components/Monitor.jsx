import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/FlameChart'

export default function Monitor({ trainTimeSeriesData, traceTimeSeriesData, traceFlameData }) {
  return (
    <div>
      {trainTimeSeriesData && <TimeSeriesChart data={trainTimeSeriesData} />}
      {traceTimeSeriesData && <TimeSeriesChart data={traceTimeSeriesData} />}
      {traceFlameData && <FlameChart data={traceFlameData} />}
    </div>
  )
}
