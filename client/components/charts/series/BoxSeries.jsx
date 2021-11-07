import React, { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'
import {
  POSITIONS_PER_RECTANGLE,
  setHorizontalLineVertices,
  setRectangleVertices,
  setVerticalLineVertices,
} from '../primitives/Rectangle'

const { Color, VertexColors, BufferAttribute } = THREE
const SQUARES_PER_DATUM = 6 // rect + 4 border lines + 1 mid line

const medianStrokeColor = new Color('#333')
const minMaxStrokeColor = new Color('#333')
const whiskerStrokeColor = new Color('#666')

// Each datum in `series.summaryData` should have the numeric properties: `x1, x2, min, p25, median, p75, max`
export default React.memo(
  ({ series, dimensions, strokeWidth = 3, fillColor = '#ccc', maxNumPoints = 1_000 }) => {
    const positions = useMemo(
      () =>
        new BufferAttribute(
          new Float32Array(3 * POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * maxNumPoints),
          3
        ),
      [maxNumPoints]
    )
    const colors = useMemo(
      () =>
        new BufferAttribute(
          new Float32Array(3 * POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * maxNumPoints),
          3
        ),
      [maxNumPoints]
    )

    const ref = useRef()

    useLayoutEffect(() => {
      const geometry = ref.current
      geometry.setAttribute('position', positions)
      geometry.setAttribute('color', colors)
    }, [maxNumPoints])

    useLayoutEffect(() => {
      const { summaryData: data } = series
      if (!data?.length) return

      const { x, y, width, height } = dimensions
      const { xDomain, yDomain } = series
      const xScale = scaleLinear()
        .domain(xDomain)
        .range([x, x + width])
      const yScale = scaleLinear()
        .domain(yDomain)
        .range([y, y + height])
      const fill = new Color(fillColor)
      const ps = positions
      const cs = colors
      const sw = strokeWidth

      data.reduce((i, { x1, x2, min, p25, median, p75, max }) => {
        const left = xScale(x1)
        const right = xScale(x2)
        const xMid = left + (right - left) / 2
        const xw = Math.min(Math.max(2, right - left - 4), 10)
        const yMed = yScale(median)
        const yMinInner = yScale(p25)
        const yMaxInner = yScale(p75)
        const yMin = yScale(min) - 1
        const yMax = yScale(max) + 1

        i = setRectangleVertices(
          ps,
          cs,
          i,
          xMid - xw / 2,
          yMinInner,
          xw,
          Math.max(2, Math.abs(yMaxInner - yMinInner)),
          fill
        )
        i = setVerticalLineVertices(ps, cs, i, yMaxInner, yMax, xMid, sw, whiskerStrokeColor)
        i = setVerticalLineVertices(ps, cs, i, yMin, yMinInner, xMid, sw, whiskerStrokeColor)
        i = setHorizontalLineVertices(ps, cs, i, xMid - xw / 3, xMid + xw / 3, yMax, sw, minMaxStrokeColor)
        i = setHorizontalLineVertices(ps, cs, i, xMid - xw / 3, xMid + xw / 3, yMin, sw, minMaxStrokeColor)
        return setHorizontalLineVertices(ps, cs, i, xMid - xw / 2, xMid + xw / 2, yMed, sw, medianStrokeColor)
      }, 0)

      const geometry = ref.current
      geometry.setDrawRange(0, (data.length - 1) * SQUARES_PER_DATUM * POSITIONS_PER_RECTANGLE)
      positions.needsUpdate = true
      colors.needsUpdate = true
    })

    return (
      <mesh>
        <bufferGeometry ref={ref} />
        <meshBasicMaterial vertexColors={VertexColors} />
      </mesh>
    )
  }
)
