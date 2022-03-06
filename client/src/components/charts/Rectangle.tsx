import React from 'react'
import { Line } from '@react-three/drei'
import type { Dimensions } from './Chart'
import type { Color } from '@react-three/fiber'

interface Props {
  dimensions: Dimensions
  color: Color
  lineWidth?: number
}

export default ({ dimensions, color, lineWidth = 1 }: Props) => {
  const { x, y, width, height } = dimensions

  return (
    <Line
      points={[
        [x, y, 0],
        [x + width, y, 0],
        [x + width, y + height, 0],
        [x, y + height, 0],
        [x, y, 0],
      ]}
      color={color}
      lineWidth={lineWidth}
    />
  )
}
