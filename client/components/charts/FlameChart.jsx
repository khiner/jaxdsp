import React, { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Html } from '@react-three/drei'
import { scaleLinear } from 'd3-scale'
import Vertices, { POSITIONS_PER_RECTANGLE } from './Vertices'
import ClipArea from './ClipArea'
import { VertexColors } from 'three'
import colors from './colors'

// Example:
//    <FlameChart data={
//      xDomain: [1631772930783, 1631772941650],
//      data: [
//        { id: 'test', label: 'Test', data: [{ x1: 1631772930783, x2: 1631772941650  }]}
//      ]
//    }/>
// Note that `duration_ms` could be different than x1 - x2, since it's
// calculated using Python's more accurate `time.perf_counter`.
export default React.memo(({ data, dimensions, renderOrder = 0, fontSize = 12 }) => {
  const ref = useRef()
  const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 1_000), [])
  const [hoveringDatumId, setHoveringDatumId] = useState(undefined)

  const { data: allSeries, xDomain } = data
  const numSeries = allSeries?.length
  const { x, y, width, height } = dimensions
  const labelWidth = 130

  useLayoutEffect(() => vertices.setGeometryRef(ref), [])
  useLayoutEffect(() => {
    vertices.draw(v => {
      allSeries?.forEach(({ data, color }, i) =>
        data.forEach(({ id, x1, x2 }) => {
          const xScale = scaleLinear()
            .domain(xDomain)
            .range([x + labelWidth, x + width])
          v.rectangle(
            xScale(x1),
            y + height - (height * (i + 1)) / numSeries,
            Math.max(xScale(x2) - xScale(x1), 4), // 4px min
            height / numSeries,
            hoveringDatumId === id ? '#00FF00' : color
          )
          // onMouseOver={() => setHoveringDatumId(id)}
          // onMouseLeave={() => {
          //   if (hoveringDatumId === id) setHoveringDatumId(undefined)
          // }}
        })
      )
    })
  })

  return (
    <>
      <>
        {allSeries?.map(({ label }, i) => (
          <Html
            key={label}
            position={[x, y + height - (height * (i + 0.5)) / numSeries + fontSize, 0]}
            style={{
              width: labelWidth,
              color: colors.axis.text,
              fontSize,
              fontWeight: 'bold',
              textAlign: 'right',
              paddingRight: '1em',
            }}
          >
            {label}
          </Html>
        ))}
      </>
      <mesh renderOrder={renderOrder}>
        <bufferGeometry ref={ref} />
        <meshBasicMaterial vertexColors={VertexColors}>
          <ClipArea dimensions={dimensions} />
        </meshBasicMaterial>
      </mesh>
    </>
  )
})
