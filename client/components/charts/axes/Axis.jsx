import React, { useLayoutEffect, useMemo, useRef } from 'react'
import { VertexColors } from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import Vertices, { POSITIONS_PER_RECTANGLE } from '../util/Vertices'
import colors from '../colors'

export default React.memo(
  ({ xDomain, yDomain, dimensions, side = 'y', strokeWidth = 2, fontSize = 12, tickLength = 10 }) => {
    const ref = useRef()
    const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 100), [])
    useLayoutEffect(() => vertices.setGeometryRef(ref), [])

    const { x, y, width, height } = dimensions
    const xScale = scaleLinear().domain(xDomain).range([x, width]).nice()
    const yScale = scaleLinear().domain(yDomain).range([y, height]).nice()
    const tickFormat = yScale.tickFormat(10)
    const ticks = yScale.ticks().map(t => ({ position: yScale(t), text: tickFormat(t) }))

    const [xStart, xEnd] = xScale.range()
    const [yStart, yEnd] = yScale.range()

    useLayoutEffect(() => {
      if (ticks.length === 0) return

      vertices.draw(v =>
        ticks.forEach(({ position }) =>
          v.rectangle(
            width - tickLength,
            position - strokeWidth / 2,
            tickLength,
            strokeWidth,
            colors.axis.stroke
          )
        )
      )
    })

    return (
      <>
        {ticks.map(({ position, text }) => (
          <Html
            key={`${position}`}
            center={false}
            position={[xStart, position + fontSize / 2, 0]}
            style={{ fontSize, color: colors.axis.text }}
          >
            {text}
          </Html>
        ))}
        <mesh>
          <bufferGeometry ref={ref} />
          <meshBasicMaterial vertexColors={VertexColors} />
        </mesh>
      </>
    )
  }
)
