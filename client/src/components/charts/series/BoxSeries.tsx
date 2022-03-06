import React, { useLayoutEffect, useMemo, useRef } from 'react'
import { VertexColors } from 'three'
import { scaleLinear } from 'd3-scale'
import Vertices, { POSITIONS_PER_RECTANGLE } from '../Vertices'
import colors from '../colors'
import ClipArea from '../ClipArea'
import type { Series } from '../Chart'

const SQUARES_PER_DATUM = 6 // rect + 4 border lines + 1 mid line
const { fill, whiskerStroke, minMaxStroke } = colors.series.box

// Each datum in `series.summaryData` should have the numeric properties: `x1, x2, min, p25, median, p75, max`
export default React.memo(({ series, dimensions, strokeWidth = 2, renderOrder = 0 }: Series) => {
  const ref = useRef()
  const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * SQUARES_PER_DATUM * 1_000), [])
  useLayoutEffect(() => vertices.setGeometryRef(ref), [])

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
    const sw = strokeWidth

    vertices.draw(v => {
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

        v.rectangle(xMid - xw / 2, yMinInner, xw, Math.max(2, Math.abs(yMaxInner - yMinInner)), fill)
        v.verticalLine(yMaxInner, yMax, xMid, sw, whiskerStroke)
        v.verticalLine(yMin, yMinInner, xMid, sw, whiskerStroke)
        v.horizontalLine(xMid - xw / 3, xMid + xw / 3, yMax, sw, minMaxStroke)
        v.horizontalLine(xMid - xw / 3, xMid + xw / 3, yMin, sw, minMaxStroke)
        v.horizontalLine(xMid - xw / 2, xMid + xw / 2, yMed, sw, minMaxStroke)
      })
    })
  })

  return (
    <mesh renderOrder={renderOrder}>
      <bufferGeometry ref={ref} />
      <meshBasicMaterial vertexColors={VertexColors} opacity={0.7} transparent={true}>
        <ClipArea dimensions={dimensions} />
      </meshBasicMaterial>
    </mesh>
  )
})
