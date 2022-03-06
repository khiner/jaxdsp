import React from 'react'
import TimeSeriesChart from './charts/TimeSeriesChart'
import FlameChart from './charts/TraceChart'
import ChartContext from './charts/ChartContext'
import Axis, { AxisSide } from './charts/Axis'
import { Data } from './charts/Chart'

const [chartWidth, flameChartHeight, xAxisHeight, yAxisWidth] = [600, 100, 40, 100]

const hasData = (data?: Data) => data?.allSeries?.length > 0

interface Props {
  trainTimeSeriesData?: Data
  traceTimeSeriesData?: Data
  traceFlameData?: Data
  width?: number
}

export default function Monitor({
  trainTimeSeriesData,
  traceTimeSeriesData,
  traceFlameData,
  width = chartWidth,
}: Props) {
  // Same time domain is shared across all time-series.
  const xDomain = trainTimeSeriesData?.xDomain || traceTimeSeriesData?.xDomain || traceFlameData?.xDomain
  if (!xDomain) return null

  console.log(xDomain)
  return (
    <ChartContext width={width}>
      {hasData(trainTimeSeriesData) && (
        <TimeSeriesChart
          title="Train"
          data={trainTimeSeriesData}
          axes={[AxisSide.left]}
          yAxisWidth={yAxisWidth}
        />
      )}
      {hasData(traceTimeSeriesData) && (
        <TimeSeriesChart
          title="Trace"
          data={traceTimeSeriesData}
          axes={[AxisSide.left]}
          yAxisWidth={yAxisWidth}
        />
      )}
      {hasData(traceFlameData) && (
        <FlameChart
          title="Methods"
          data={traceFlameData}
          dimensions={{ height: flameChartHeight }}
          yAxisWidth={yAxisWidth}
        />
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
