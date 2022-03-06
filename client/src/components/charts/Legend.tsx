import React from 'react'
import { Html } from '@react-three/drei'
import { Dimensions, Series } from './Chart'
import colors from './ChartColors'

interface Props {
  parentDimensions: Dimensions
  allSeries: Series[]
}

export default function Legend({ parentDimensions, allSeries }: Props) {
  const { x, y, width, height } = parentDimensions

  return (
    <Html
      position={[x + width, y + height, 0]}
      style={{
        fontSize: 10,
        color: colors.text,
        position: 'absolute',
        right: 0,
        border: '1px solid #4449',
        borderRadius: 4,
        backgroundColor: '#eee9',
        padding: 2,
        margin: 4,
        marginLeft: -6,
      }}
    >
      {allSeries.map(({ id, color, label }) => (
        <div
          key={id}
          style={{ display: 'flex', flexDirection: 'row', textAlign: 'center', alignItems: 'center' }}
        >
          <span style={{ width: 10, height: 4, backgroundColor: color, margin: 4 }} />
          <span>{label}</span>
        </div>
      ))}
    </Html>
  )
}
