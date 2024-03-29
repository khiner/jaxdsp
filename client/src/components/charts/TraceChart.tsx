import React, { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Html } from '@react-three/drei'
import { scaleLinear } from 'd3-scale'
import Vertices, { POSITIONS_PER_RECTANGLE } from './Vertices'
import ClipArea from './ClipArea'
import { VertexColors } from 'three'
import colors from './ChartColors'
import type Chart from './Chart'
import Title, { DEFAULT_TITLE_HEIGHT } from './Title'

export default React.memo(
  ({ title, data, dimensions, renderOrder = 0, fontSize = 12, yAxisWidth = 60, paddingTop = 12 }: Chart) => {
    const ref = useRef()
    const vertices = useMemo(() => new Vertices(POSITIONS_PER_RECTANGLE * 1_000), [])
    const [hoveringDatumId, setHoveringDatumId] = useState(undefined)

    const { allSeries: allSeries, xDomain } = data
    const numSeries = allSeries?.length
    const { x, y, width, height } = dimensions
    const allSeriesHeight = height - DEFAULT_TITLE_HEIGHT - paddingTop
    const seriesHeight = allSeriesHeight / numSeries
    useLayoutEffect(() => vertices.setGeometryRef(ref), [])
    useLayoutEffect(() => {
      vertices.draw(v => {
        allSeries?.forEach(({ id: seriesId, data, color }, i) =>
          data.forEach(({ x1, x2 }) => {
            const id = `${seriesId}-${x1}`
            const xScale = scaleLinear()
              .domain(xDomain)
              .range([x + yAxisWidth, x + width])
            v.rectangle(
              xScale(x1),
              y + allSeriesHeight - (i + 1) * seriesHeight,
              xScale(x2) - xScale(x1),
              seriesHeight,
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
        {!!title && (
          <Title
            title={title}
            dimensions={{ x: x + yAxisWidth, y: y + allSeriesHeight, width: width - yAxisWidth }}
          />
        )}
        <>
          {allSeries?.map(({ id, label }, i) => (
            <Html
              key={id}
              position={[x, y + allSeriesHeight - (i + 0.5) * seriesHeight + fontSize, 0]}
              style={{
                width: yAxisWidth,
                color: colors.text,
                fontSize,
                fontWeight: 'bold',
                textAlign: 'right',
                paddingRight: '1em',
                overflowWrap: 'break-word',
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
  }
)
