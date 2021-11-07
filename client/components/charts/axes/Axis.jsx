import React, { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { BufferAttribute } from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import { setRectangleVertices, POSITIONS_PER_RECTANGLE } from '../primitives/Rectangle'

const { Color, VertexColors } = THREE

export default React.memo(
  ({
    xDomain,
    yDomain,
    dimensions,
    side = 'y',
    strokeWidth = 2,
    strokeColor = '#333',
    textColor = '#333',
    fontSize = 12,
    tickLength = 10,
    maxNumPoints = 100,
  }) => {
    const ref = useRef()
    const positions = useMemo(
      () => new BufferAttribute(new Float32Array(3 * POSITIONS_PER_RECTANGLE * maxNumPoints), 3),
      [maxNumPoints]
    )
    const colors = useMemo(
      () => new BufferAttribute(new Float32Array(3 * POSITIONS_PER_RECTANGLE * maxNumPoints), 3),
      [maxNumPoints]
    )

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
      const geometry = ref.current
      geometry.setAttribute('position', positions)
      geometry.setAttribute('color', colors)
    }, [maxNumPoints])

    useLayoutEffect(() => {
      if (ticks.length === 0) return

      const strokeFill = new Color(strokeColor)
      ticks.reduce(
        (i, { position }) =>
          setRectangleVertices(
            positions,
            colors,
            i,
            xStart + 40,
            position - strokeWidth,
            tickLength,
            strokeWidth,
            strokeFill
          ),
        0
      )

      const geometry = ref.current
      geometry.setDrawRange(0, (ticks.length - 1) * POSITIONS_PER_RECTANGLE)
      positions.needsUpdate = true
      colors.needsUpdate = true
    })

    return (
      <>
        {ticks.map(({ position, text }) => (
          <Html
            key={`${position}`}
            center={false}
            position={[xStart, position + fontSize / 2, 0]}
            style={{ fontSize }}
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
