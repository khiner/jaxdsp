import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'
import {
  addHorizontalLineVertices,
  addRectangleVertices,
  addVerticalLineVertices,
  POSITIONS_PER_RECTANGLE,
  VERTICES_PER_POSITION,
} from '../primitives/Rectangle'
import { useThree } from '@react-three/fiber'

const { Color, VertexColors, Float32BufferAttribute } = THREE
const SQUARES_PER_DATUM = 6 // rect + 4 border lines + 1 mid line

const medianStrokeColor = new Color('#333')
const minMaxStrokeColor = new Color('#333')
const whiskerStrokeColor = new Color('#666')

// Each datum in `series.summaryData` should have the numeric properties: `x1, x2, min, p25, median, p75, max`
export default React.memo(({ series, strokeWidth = 3, fillColor = '#ccc', maxNumPoints = 10_000 }) => {
  const [positions, setPositions] = useState(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))
  const [colors, setColors] = useState(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))

  const ref = useRef()
  const { size } = useThree()
  const { width, height } = size

  useEffect(() => {
    setPositions(
      new Float32Array(VERTICES_PER_POSITION * POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * maxNumPoints)
    )
    setColors(
      new Float32Array(VERTICES_PER_POSITION * POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * maxNumPoints)
    )
  }, [maxNumPoints])

  useLayoutEffect(() => {
    const { summaryData: data } = series
    if (!data?.length) return

    const { xDomain, yDomain } = series
    const xScale = scaleLinear().domain(xDomain).range([0, width])
    const yScale = scaleLinear().domain(yDomain).range([0, height])
    const fill = new Color(fillColor)

    let i = 0
    data.forEach(datum => {
      const { x1, x2, min, p25, median, p75, max } = datum
      const left = xScale(x1)
      const right = xScale(x2)
      const xMid = left + (right - left) / 2
      const xw = Math.min(Math.max(2, right - left - 4), 10)
      const yMed = yScale(median)
      const yMinInner = yScale(p25)
      const yMaxInner = yScale(p75)
      const yMin = yScale(min) - 1
      const yMax = yScale(max) + 1
      const ps = positions
      const cs = colors

      i = addRectangleVertices(
        ps,
        cs,
        i,
        xMid - xw / 2,
        yMinInner,
        xw,
        Math.max(2, Math.abs(yMaxInner - yMinInner)),
        fill
      )
      const sw = strokeWidth
      i = addVerticalLineVertices(ps, cs, i, yMaxInner, yMax, xMid, sw, whiskerStrokeColor)
      i = addVerticalLineVertices(ps, cs, i, yMin, yMinInner, xMid, sw, whiskerStrokeColor)
      i = addHorizontalLineVertices(ps, cs, i, xMid - xw / 3, xMid + xw / 3, yMax, sw, minMaxStrokeColor)
      i = addHorizontalLineVertices(ps, cs, i, xMid - xw / 3, xMid + xw / 3, yMin, sw, minMaxStrokeColor)
      i = addHorizontalLineVertices(ps, cs, i, xMid - xw / 2, xMid + xw / 2, yMed, sw, medianStrokeColor)
    })

    const geometry = ref.current
    geometry.setAttribute('position', new Float32BufferAttribute(positions, VERTICES_PER_POSITION))
    geometry.setAttribute('color', new Float32BufferAttribute(colors, VERTICES_PER_POSITION))
    geometry.setDrawRange(0, (data.length - 1) * SQUARES_PER_DATUM * POSITIONS_PER_RECTANGLE)
  })

  return (
    <mesh>
      <bufferGeometry ref={ref} />
      <meshBasicMaterial vertexColors={VertexColors} />
    </mesh>
  )
})
