import React from 'react'
import { Line } from '@react-three/drei'

export default ({ dimensions, color, lineWidth = 1 }) => {
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
