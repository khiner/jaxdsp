import React, { useLayoutEffect, useRef } from 'react'
import { useThree } from '@react-three/fiber'

import FatLine from './primitives/FatLine'

const MAX_NUM_POINTS = 100_000
const positions = new Float32Array(3 * MAX_NUM_POINTS)

export default React.memo(({ data, strokeWidth = 2, strokeColor = '#666' }) => {
  const { size } = useThree()
  const { width, height } = size

  // TODO track min/max in each series in accumulator
  const xs = data.map(({ x }) => x)
  const ys = data.map(({ y }) => y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const xRange = maxX - minX
  const yRange = maxY - minY

  const ref = useRef()
  useLayoutEffect(() => {
    const geom = ref.current
    data.forEach((d, i) => {
      positions[i * 3] = (width * (d.x - minX)) / xRange
      positions[i * 3 + 1] = (height * (d.y - minY)) / yRange
      positions[i * 3 + 2] = 0
    })
    geom.setPositions(positions)
    geom.instanceCount = data.length - 1
  }, [data])

  return (
    <FatLine ref={ref} width={width} height={height} strokeWidth={strokeWidth} strokeColor={strokeColor} />
  )
})
