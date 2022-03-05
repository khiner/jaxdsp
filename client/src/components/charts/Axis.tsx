import React, { useLayoutEffect, useMemo, useRef } from 'react'
import { VertexColors } from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import Vertices, { POSITIONS_PER_RECTANGLE } from './Vertices'
import colors from './colors'
import { timeFormat } from 'd3-time-format'
import type { Dimensions, Domain } from './series/Series'

const formatMinutesSeconds = timeFormat('%M:%S')

export enum AxisSide {
  left,
  bottom,
}

const createTicks = (side, xDomain: Domain, yDomain: Domain, { x, y, width, height }: Dimensions) => {
  if (side === AxisSide.left) {
    const yScale = scaleLinear()
      .domain(yDomain)
      .range([y, y + height])
      .nice()
    const tickFormat = yScale.tickFormat(10)
    return yScale.ticks().map(t => ({ position: yScale(t), text: tickFormat(t) }))
  }

  const xScale = scaleLinear()
    .domain(xDomain)
    .range([x, x + width])
  return xScale.ticks().map(t => ({ position: xScale(t), text: formatMinutesSeconds(t) }))
}

interface Props {
  dimensions: Dimensions
  side: AxisSide
  xDomain?: Domain
  yDomain?: Domain
  strokeWidth?: number
  fontSize?: number
  tickLength?: number
}

export default React.memo(
  ({
    xDomain,
    yDomain,
    dimensions,
    side = AxisSide.left,
    strokeWidth = 2,
    fontSize = 12,
    tickLength = 10,
  }: Props) => {
    const ref = useRef()
    const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 100), [])
    useLayoutEffect(() => vertices.setGeometryRef(ref), [])

    const isLeft = side === AxisSide.left
    const { x, y, width, height } = dimensions
    const ticks = createTicks(side, xDomain, yDomain, dimensions)

    useLayoutEffect(() => {
      if (ticks.length === 0) return

      vertices.draw(v =>
        ticks.forEach(({ position }) =>
          isLeft
            ? v.rectangle(
                x + width - tickLength,
                position - strokeWidth / 2,
                tickLength,
                strokeWidth,
                colors.axis.stroke
              )
            : v.rectangle(
                position - strokeWidth / 2,
                y + height - tickLength,
                strokeWidth,
                tickLength,
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
            center={side === AxisSide.bottom}
            position={
              isLeft ? [x, position + (3 * fontSize) / 4, 0] : [position, y + height - (3 * fontSize) / 2, 0]
            }
            style={{
              fontSize,
              color: colors.axis.text,
              textAlign: isLeft ? 'right' : undefined,
              width: isLeft ? width - tickLength : undefined,
              paddingRight: isLeft ? '1em' : undefined,
            }}
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
