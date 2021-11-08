import React, { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import Vertices, { POSITIONS_PER_RECTANGLE } from '../util/Vertices'
import colors from '../colors'

const { VertexColors } = THREE

export default React.memo(
  ({
    xDomain,
    yDomain,
    dimensions,
    side = 'y',
    strokeWidth = 2,
    fontSize = 12,
    tickLength = 10,
    maxLength = 100,
  }) => {
    const ref = useRef()
    const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * maxLength), [maxLength])
    useLayoutEffect(() => vertices.setGeometryRef(ref), [maxLength])

    const { x, y, width, height } = dimensions
    const xScale = scaleLinear().domain(xDomain).range([x, width])
    const yScale = scaleLinear().domain(yDomain).range([y, height])
    const tickFormat = yScale.tickFormat(10)
    const ticks = yScale.ticks().map(t => ({
      position: yScale(t),
      text: tickFormat(t),
    }))

    const [xStart, xEnd] = xScale.range()
    const [yStart, yEnd] = yScale.range()

    useLayoutEffect(() => {
      if (ticks.length === 0) return

      vertices.draw(v => {
        v.rectangle(x, y, width, height, colors.background)
        ticks.forEach(({ position }) =>
          v.rectangle(xStart + 40, position - strokeWidth, tickLength, strokeWidth, colors.axis.stroke)
        )
      })
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
