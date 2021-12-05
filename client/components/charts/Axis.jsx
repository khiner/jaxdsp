import React, { useLayoutEffect, useMemo, useRef } from 'react'
import { VertexColors } from 'three'
import { scaleLinear } from 'd3-scale'
import { Html } from '@react-three/drei'
import Vertices, { POSITIONS_PER_RECTANGLE } from './Vertices'
import colors from './colors'
import { timeFormat } from 'd3-time-format'

const formatMinutesSeconds = timeFormat('%M:%S')

// Currently assumes `side` is either 'left' or 'bottom'
export default React.memo(
  ({ xDomain, yDomain, dimensions, side = 'left', strokeWidth = 2, fontSize = 12, tickLength = 10 }) => {
    const ref = useRef()
    const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 100), [])
    useLayoutEffect(() => vertices.setGeometryRef(ref), [])

    const { x, y, width, height } = dimensions
    const xScale = scaleLinear()
      .domain(xDomain)
      .range([x, x + width])
      .nice()
    const yScale = scaleLinear()
      .domain(yDomain)
      .range([y, y + height])
      .nice()
    const yTickFormat = yScale.tickFormat(10)
    const ticks =
      side === 'left'
        ? yScale.ticks().map(t => ({ position: yScale(t), text: yTickFormat(t) }))
        : xScale.ticks().map(t => ({ position: xScale(t), text: formatMinutesSeconds(t) }))

    useLayoutEffect(() => {
      if (ticks.length === 0) return

      vertices.draw(v =>
        ticks.forEach(({ position }) =>
          side === 'left'
            ? v.rectangle(
                width - tickLength,
                position - strokeWidth / 2,
                tickLength,
                strokeWidth,
                colors.axis.stroke
              )
            : v.rectangle(
                position - strokeWidth / 2,
                height - tickLength,
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
            center={false}
            position={
              side === 'left' ? [x, position + fontSize / 2, 0] : [position, height - fontSize / 2, 0]
            }
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
