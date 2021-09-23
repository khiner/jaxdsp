import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Float32BufferAttribute } from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import { addRectangleVertices, POSITIONS_PER_RECTANGLE, VERTICES_PER_POSITION } from '../primitives/Rectangle'

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
    maxNumPoints = 1_000,
  }) => {
    const ref = useRef()
    const [positions, setPositions] = useState(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))
    const [colors, setColors] = useState(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))

    useEffect(() => {
      setPositions(new Float32Array(VERTICES_PER_POSITION * POSITIONS_PER_RECTANGLE * maxNumPoints))
      setColors(new Float32Array(VERTICES_PER_POSITION * POSITIONS_PER_RECTANGLE * maxNumPoints))
    }, [maxNumPoints])

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

      const strokeFill = new Color(strokeColor)
      let i = 0
      ticks.forEach(({ position }) => {
        i = addRectangleVertices(
          positions,
          colors,
          i,
          xStart + 40,
          position - strokeWidth,
          tickLength,
          strokeWidth,
          strokeFill
        )
      })

      const geometry = ref.current
      geometry.setAttribute('position', new Float32BufferAttribute(positions, VERTICES_PER_POSITION))
      geometry.setAttribute('color', new Float32BufferAttribute(colors, VERTICES_PER_POSITION))
      geometry.setDrawRange(0, (ticks.length - 1) * POSITIONS_PER_RECTANGLE)
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
