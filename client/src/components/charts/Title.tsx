import { Html } from '@react-three/drei'
import colors from './ChartColors'
import React from 'react'
import { Dimensions } from './Chart'

export const DEFAULT_TITLE_HEIGHT = 24

interface Props {
  title: string
  dimensions: Dimensions
}

export default function Title({ title, dimensions }: Props) {
  const { x, y, width, height = DEFAULT_TITLE_HEIGHT } = dimensions

  return (
    <Html
      position={[x, y + height, 0]}
      style={{
        fontSize: 14,
        color: colors.text,
        width,
        height,
        textAlign: 'center',
      }}
    >
      {title}
    </Html>
  )
}
