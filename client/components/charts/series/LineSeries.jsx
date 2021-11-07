import React from 'react'

import { scaleLinear } from 'd3-scale'
import { setPosition, VERTICES_PER_POSITION } from '../primitives/Rectangle'
import { Line } from '@react-three/drei'

export default React.memo(({ series, dimensions, strokeWidth = 3, strokeColor = '#666' }) => {
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

  const positions = new Array(data.length * VERTICES_PER_POSITION)
  data.reduce((i, { x, y }) => setPosition(positions, undefined, i, xScale(x), yScale(y)), 0)

  return <Line points={positions} lineWidth={strokeWidth} color={strokeColor} />
})
