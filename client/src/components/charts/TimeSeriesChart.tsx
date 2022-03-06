import React from 'react'
import LineSeries from './series/LineSeries'
import BoxSeries from './series/BoxSeries'
import Axis, { AxisSide } from './Axis'
import colors from './ChartColors'
import Rectangle from './Rectangle'
import type Chart from './Chart'
import GridLines from './GridLines'
import Title, { DEFAULT_TITLE_HEIGHT } from './Title'

// TODO show points for start/end of contiguous ranges
export default React.memo(
  ({
    title,
    data,
    dimensions,
    axes = [AxisSide.left, AxisSide.bottom],
    xAxisHeight = 40,
    yAxisWidth = 60,
    paddingTop = 12,
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
      height: height - xAxisHeight - DEFAULT_TITLE_HEIGHT - paddingTop,
    }

    return (
      <>
        {!!title && (
          <Title
            title={title}
            dimensions={{
              x: seriesDimensions.x,
              y: y + seriesDimensions.height,
              width: seriesDimensions.width,
            }}
          />
        )}
        {grid && (
          <GridLines dimensions={seriesDimensions} xDomain={xDomain} yDomain={yDomain} renderOrder={-2} />
        )}
        {allSeries.map(series => (
          <LineSeries
            key={series.id}
            series={series}
            dimensions={seriesDimensions}
            xDomain={xDomain}
            yDomain={yDomain}
            renderOrder={-1}
          />
        ))}
        {allSeries.map(series => (
          <BoxSeries
            key={series.id}
            series={series}
            dimensions={seriesDimensions}
            xDomain={xDomain}
            yDomain={yDomain}
            renderOrder={1}
          />
        ))}
        <Rectangle dimensions={seriesDimensions} color={colors.border} />
        {hasLeftAxis && (
          <Axis
            side={AxisSide.left}
            yDomain={yDomain}
            dimensions={{ x, y: y + xAxisHeight, width: yAxisWidth, height: seriesDimensions.height }}
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
