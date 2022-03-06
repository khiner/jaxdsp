import React, { useLayoutEffect, useMemo, useRef } from 'react'
import { VertexColors } from 'three'
import Vertices, { POSITIONS_PER_RECTANGLE } from './Vertices'
import colors from './ChartColors'
import type { Dimensions, Domain } from './Chart'
import { AxisSide, createTicks } from './Axis'

interface Props {
  dimensions: Dimensions
  xDomain?: Domain
  yDomain?: Domain
  strokeWidth?: number
  renderOrder?: number
}

export default React.memo(({ dimensions, xDomain, yDomain, strokeWidth = 1, renderOrder = 0 }: Props) => {
  const ref = useRef()
  const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 100), [])
  useLayoutEffect(() => vertices.setGeometryRef(ref), [])

  const { x, y, width, height } = dimensions
  const horizontalTicks = createTicks(AxisSide.left, dimensions, yDomain, xDomain)
  const verticalTicks = createTicks(AxisSide.bottom, dimensions, yDomain, xDomain)

  useLayoutEffect(() => {
    if (horizontalTicks.length === 0 || verticalTicks.length === 0) return

    vertices.draw(v => {
      horizontalTicks.forEach(({ position }) =>
        v.rectangle(x, position - strokeWidth / 2, width, strokeWidth, colors.grid.stroke)
      )
      verticalTicks.forEach(({ position }) =>
        v.rectangle(position - strokeWidth / 2, y, strokeWidth, height, colors.grid.stroke)
      )
    })
  })

  return (
    <mesh renderOrder={renderOrder}>
      <bufferGeometry ref={ref} />
      <meshBasicMaterial vertexColors={VertexColors} />
    </mesh>
  )
})
