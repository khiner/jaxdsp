import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/TraceChart'
import ChartContext from './charts/ChartContext'
import Axis, { AxisSide } from './charts/Axis'

const [chartWidth, flameChartHeight, xAxisHeight, yAxisWidth] = [600, 100, 40, 100]

const hasData = data => data?.data?.length > 0

export default function Monitor({
  width = chartWidth,
  trainTimeSeriesData,
  traceTimeSeriesData,
  traceFlameData,
}) {
  if (!trainTimeSeriesData && !traceTimeSeriesData && !traceFlameData) return null

  const { xDomain } = traceTimeSeriesData // time domain shared across all time-series
  return (
    <ChartContext width={width}>
      {hasData(trainTimeSeriesData) && (
        <TimeSeriesChart data={trainTimeSeriesData} axes={[AxisSide.left]} yAxisWidth={yAxisWidth} />
      )}
      {hasData(traceTimeSeriesData) && (
        <TimeSeriesChart data={traceTimeSeriesData} axes={[AxisSide.left]} yAxisWidth={yAxisWidth} />
      )}
      {hasData(traceFlameData) && (
        <FlameChart data={traceFlameData} dimensions={{ height: flameChartHeight }} yAxisWidth={yAxisWidth} />
      )}
      {xDomain && (
        <Axis
          side={AxisSide.bottom}
          xDomain={xDomain}
          dimensions={{ x: yAxisWidth, width: width - yAxisWidth, height: xAxisHeight }}
        />
      )}
    </ChartContext>
  )
}
