import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/TraceChart'
import ChartContext from './charts/ChartContext'
import Axis, { BOTTOM, LEFT } from './charts/Axis'

const [chartWidth, chartHeight, flameChartHeight, xAxisHeight, yAxisWidth] = [600, 200, 100, 40, 100]

export default function Monitor({
  width = chartWidth,
  height = chartHeight * 2 + flameChartHeight + xAxisHeight,
  trainTimeSeriesData,
  traceTimeSeriesData,
  traceFlameData,
}) {
  if (!trainTimeSeriesData && !traceTimeSeriesData && !traceFlameData) return null

  const { xDomain } = traceTimeSeriesData // time domain shared across all time-series
  return (
    <ChartContext width={width} height={height}>
      {trainTimeSeriesData && (
        <TimeSeriesChart
          data={trainTimeSeriesData}
          dimensions={{
            x: 0,
            y: xAxisHeight + flameChartHeight + chartHeight,
            width: chartWidth,
            height: chartHeight,
          }}
          axes={[LEFT]}
          yAxisWidth={yAxisWidth}
        />
      )}
      {traceTimeSeriesData && (
        <TimeSeriesChart
          data={traceTimeSeriesData}
          dimensions={{ x: 0, y: xAxisHeight + flameChartHeight, width: chartWidth, height: chartHeight }}
          axes={[LEFT]}
          yAxisWidth={yAxisWidth}
        />
      )}
      {traceFlameData && (
        <FlameChart
          data={traceFlameData}
          dimensions={{ x: 0, y: xAxisHeight, width: chartWidth, height: flameChartHeight }}
          yAxisWidth={yAxisWidth}
        />
      )}
      {xDomain && (
        <Axis
          side={BOTTOM}
          xDomain={xDomain}
          dimensions={{ x: yAxisWidth, y: 0, width, height: xAxisHeight }}
        />
      )}
    </ChartContext>
  )
}
