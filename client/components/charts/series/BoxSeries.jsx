import React, { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'
import Vertices, { POSITIONS_PER_RECTANGLE } from '../util/Vertices'

const { Color, VertexColors } = THREE
const SQUARES_PER_DATUM = 6 // rect + 4 border lines + 1 mid line

const medianStrokeColor = new Color('#333')
const minMaxStrokeColor = new Color('#333')
const whiskerStrokeColor = new Color('#666')

// Each datum in `series.summaryData` should have the numeric properties: `x1, x2, min, p25, median, p75, max`
export default React.memo(
  ({ series, dimensions, strokeWidth = 3, fillColor = '#ccc', maxLength = 1_000 }) => {
    const ref = useRef()
    const vertices = useMemo(
      () => new Vertices(POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * maxLength),
      [maxLength]
    )
    useLayoutEffect(() => vertices.setGeometryRef(ref), [maxLength])

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
      const sw = strokeWidth

      vertices.start()
      data.forEach(({ x1, x2, min, p25, median, p75, max }) => {
        const left = xScale(x1)
        const right = xScale(x2)
        const xMid = left + (right - left) / 2
        const xw = Math.min(Math.max(2, right - left - 4), 10)
        const yMed = yScale(median)
        const yMinInner = yScale(p25)
        const yMaxInner = yScale(p75)
        const yMin = yScale(min) - 1
        const yMax = yScale(max) + 1

        vertices.addRectangle(
          xMid - xw / 2,
          yMinInner,
          xw,
          Math.max(2, Math.abs(yMaxInner - yMinInner)),
          fill
        )
        vertices.addVerticalLine(yMaxInner, yMax, xMid, sw, whiskerStrokeColor)
        vertices.addVerticalLine(yMin, yMinInner, xMid, sw, whiskerStrokeColor)
        vertices.addHorizontalLine(xMid - xw / 3, xMid + xw / 3, yMax, sw, minMaxStrokeColor)
        vertices.addHorizontalLine(xMid - xw / 3, xMid + xw / 3, yMin, sw, minMaxStrokeColor)
        vertices.addHorizontalLine(xMid - xw / 2, xMid + xw / 2, yMed, sw, medianStrokeColor)
      })
      vertices.end()
    })

    return (
      <mesh>
        <bufferGeometry ref={ref} />
        <meshBasicMaterial vertexColors={VertexColors} />
      </mesh>
    )
  }
)
