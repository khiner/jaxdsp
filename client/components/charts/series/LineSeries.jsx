import React from 'react'

import { scaleLinear } from 'd3-scale'
import { setPosition, VERTICES_PER_POSITION } from '../primitives/Rectangle'
import { Line } from '@react-three/drei'

export default React.memo(({ series, dimensions, strokeWidth = 3, strokeColor = '#666' }) => {
  const { x, y, width, height } = dimensions

  const { data } = series
  if (!data?.length) return null

  const positions = new Array(data.length * VERTICES_PER_POSITION)
  const { xDomain, yDomain } = series
  const xScale = scaleLinear()
    .domain(xDomain)
    .range([x, x + width])
  const yScale = scaleLinear()
    .domain(yDomain)
    .range([y, y + height])

  let i = 0
  data.forEach(({ x, y }) => {
    i = setPosition(positions, undefined, i, xScale(x), yScale(y))
  })

  return <Line points={positions} lineWidth={strokeWidth} color={strokeColor} />
})
