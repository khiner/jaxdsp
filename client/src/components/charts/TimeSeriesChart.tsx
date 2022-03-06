import React from 'react'
import LineSeries from './series/LineSeries'
import BoxSeries from './series/BoxSeries'
import Axis, { AxisSide } from './Axis'
import colors from './ChartColors'
import Rectangle from './Rectangle'
import type Chart from './Chart'
import GridLines from './GridLines'

// TODO show points for start/end of contiguous ranges
export default React.memo(
  ({
    data,
    dimensions,
    axes = [AxisSide.left, AxisSide.bottom],
    xAxisHeight = 40,
    yAxisWidth = 60,
    grid = true,
  }: Chart) => {
    if (!data) return null
    const { data: allSeries, xDomain, yDomain } = data
    if (!allSeries?.length) return null

    const { x, y, width, height } = dimensions
    const hasLeftAxis = axes.includes(AxisSide.left)
    const hasBottomAxis = axes.includes(AxisSide.bottom)

    if (!hasLeftAxis) yAxisWidth = 0
    if (!hasBottomAxis) xAxisHeight = 0

    const seriesDimensions = {
      x: x + yAxisWidth,
      y: y + xAxisHeight,
      width: width - yAxisWidth,
      height: height - xAxisHeight,
    }

    return (
      <>
        {grid && (
          <GridLines dimensions={seriesDimensions} xDomain={xDomain} yDomain={yDomain} renderOrder={-2} />
        )}
        {allSeries.map(series => (
          <LineSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={-1} />
        ))}
        {allSeries.map(series => (
          <BoxSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={1} />
        ))}
        {/*{allSeries.map(series => (*/}
        {/*  <ScatterSeries key={series.id} series={series} dimensions={seriesDimensions} renderOrder={2} />*/}
        {/*))}*/}
        <Rectangle dimensions={seriesDimensions} color={colors.border} />
        {hasLeftAxis && (
          <Axis
            side={AxisSide.left}
            yDomain={yDomain}
            dimensions={{ x, y: y + xAxisHeight, width: yAxisWidth, height: height - xAxisHeight }}
          />
        )}
        {hasBottomAxis && (
          <Axis
            side={AxisSide.bottom}
            xDomain={xDomain}
            dimensions={{ x: x + yAxisWidth, y, width, height: xAxisHeight }}
          />
        )}
      </>
    )
  }
)
