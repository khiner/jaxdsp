import React, { useLayoutEffect, useRef } from 'react'
import { useThree } from '@react-three/fiber'

import FatLine from './primitives/FatLine'
import { scaleLinear } from 'd3-scale'

const MAX_NUM_POINTS = 100_000
const positions = new Float32Array(3 * MAX_NUM_POINTS)

export default React.memo(({ series, strokeWidth = 2, strokeColor = '#666' }) => {
  const ref = useRef()
  const { size } = useThree()
  const { width, height } = size

  useLayoutEffect(() => {
    const { data } = series
    if (!data?.length) return

    const { xDomain, yDomain } = series
    const xScale = scaleLinear().domain(xDomain).range([0, width])
    const yScale = scaleLinear().domain(yDomain).range([0, height])

    const geom = ref.current
    data.forEach(({ x, y }, i) => {
      positions[i * 3] = xScale(x)
      positions[i * 3 + 1] = yScale(y)
      positions[i * 3 + 2] = 0
    })
    geom.setPositions(positions)
    geom.instanceCount = data.length - 1
  })

  return (
    <FatLine ref={ref} width={width} height={height} strokeWidth={strokeWidth} strokeColor={strokeColor} />
  )
})
