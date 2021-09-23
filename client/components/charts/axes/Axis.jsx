import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Float32BufferAttribute } from 'three'
import { scaleLinear } from 'd3-scale'
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

    useLayoutEffect(() => {
      const xScale = scaleLinear().domain(xDomain).range([x, width])
      const yScale = scaleLinear().domain(yDomain).range([y, height])

      const [xStart, xEnd] = xScale.range()
      const [yStart, yEnd] = yScale.range()
      const ticks = yScale.ticks()
      if (ticks.length === 0) return

      const strokeFill = new Color(strokeColor)
      let i = 0
      ticks.forEach(t => {
        const y = yScale(t)
        const tickLength = 10
        i = addRectangleVertices(positions, colors, i, xStart, y, tickLength, strokeWidth, strokeFill)
      })

      const geometry = ref.current
      geometry.setAttribute('position', new Float32BufferAttribute(positions, VERTICES_PER_POSITION))
      geometry.setAttribute('color', new Float32BufferAttribute(colors, VERTICES_PER_POSITION))
      geometry.setDrawRange(0, (ticks.length - 1) * POSITIONS_PER_RECTANGLE)
    })

    return (
      <mesh>
        <bufferGeometry ref={ref} />
        <meshBasicMaterial vertexColors={VertexColors} />
      </mesh>
    )
  }
)
