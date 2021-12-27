import React from 'react'
import { scaleLinear } from 'd3-scale'
import { Line } from '@react-three/drei'
import colors from '../colors'

export default React.memo(({ series, dimensions, strokeWidth = 2, renderOrder = 0 }) => {
  const { data } = series
  if (!data?.length) return null

  const { x, y, width, height } = dimensions
  const { xDomain, yDomain } = series
  const xScale = scaleLinear()
    .domain(xDomain)
    .range([x, x + width])
  const yScale = scaleLinear()
    .domain(yDomain)
    .range([y, y + height])

  const positions = new Array(data.length * 3)
  data.forEach(({ x, y }, i) => {
    positions[i * 3] = xScale(x)
    positions[i * 3 + 1] = yScale(y)
    positions[i * 3 + 2] = 0
  })

  return (
    <Line
      points={positions}
      lineWidth={strokeWidth}
      color={colors.series.line.stroke}
      renderOrder={renderOrder}
    />
  )
})
