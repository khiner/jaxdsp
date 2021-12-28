import React from 'react'
import LineSeries from './series/LineSeries'
import BoxSeries from './series/BoxSeries'
import ScatterSeries from './series/ScatterSeries'
import Axis from './Axis'
import colors from './colors'
import Rectangle from './Rectangle'

// `data` is a list of with `x` values assumed to be milliseconds since epoch.
// Example:
//    <TimeSeriesChart data={{
//      xDomain: [1631772930783, 1631772930783],
//      yDomain: [0.01, 0.01],
//      data: [{ id: 'test', label: 'Test', data: [{ x: 1631772930783, y: 0.01 }]}]
//    }/>
// TODO show points for start/end of contiguous ranges
export default React.memo(({ data, dimensions }) => {
  if (!data) return null
  const { data: allSeries } = data
  if (!allSeries?.length) return null

  const { xDomain, yDomain } = data

  const { x, y, width, height } = dimensions
  const xAxisHeight = 40
  const yAxisWidth = 60
  const seriesDimensions = {
    x: x + yAxisWidth,
    y: y + xAxisHeight,
    width: width - yAxisWidth,
    height: height - xAxisHeight,
  }
  const leftAxisDimensions = { x, y: y + xAxisHeight, width: yAxisWidth, height: height - xAxisHeight }
  const bottomAxisDimensions = {
    x: x + yAxisWidth,
    y,
    width: width,
    height: xAxisHeight,
  }

  return (
    <>
      {allSeries.map(series => (
        <LineSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={-1} />
      ))}
      {allSeries.map(series => (
        <BoxSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={1} />
      ))}
      {allSeries.map(series => (
        <ScatterSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={2} />
      ))}
      <Rectangle dimensions={seriesDimensions} color={colors.border} />
      <Axis side="left" xDomain={xDomain} yDomain={yDomain} dimensions={leftAxisDimensions} />
      <Axis side="bottom" xDomain={xDomain} yDomain={yDomain} dimensions={bottomAxisDimensions} />
    </>
  )
})
