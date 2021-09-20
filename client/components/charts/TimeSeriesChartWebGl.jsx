import React from 'react'
import { Canvas } from '@react-three/fiber'
import Line from './webgl/Line'
import { timeFormat } from 'd3-time-format'
import { scaleLinear } from 'd3-scale'

const formatMinutesSeconds = timeFormat('%M:%S')

// `data` is a list of with `x` values assumed to be milliseconds since epoch.
// Example:
//    <TimeSeriesChart data={{
//      xDomain: [1631772930783, 1631772930783],
//      yDomain: [0.01, 0.01],
//      data: [{ id: 'test', label: 'Test', data: [{ x: 1631772930783, y: 0.01 }]}]
//    }/>
// TODO show points for start/end of contiguous ranges
export default React.memo(({ data, width = 400, height = 200 }) => {
  if (!data) return null
  const { data: allSeries } = data
  if (!allSeries?.length) return null

  const { xDomain, yDomain } = data
  const xScale = scaleLinear().domain(xDomain).range([0, width])
  const yScale = scaleLinear().domain(yDomain).range([0, height])
  const [xStart, xEnd] = xScale.range()
  const [yStart, yEnd] = yScale.range()
  const ticks = xScale.ticks()

  return (
    <Canvas
      style={{ width, height }}
      onCreated={({ camera }) => {
        // Calculate camera z so that the top and bottom are exactly at the edges of the fov
        // Based on https://stackoverflow.com/a/13351534/780425
        // Adding height for extra space to not clip horizontal lines exactly at 0/height in half.
        const maxLineWidth = 4
        const z = (height + maxLineWidth) / (2 * Math.tan((camera.fov / 360) * Math.PI))
        camera.position.set(width / 2, height / 2, z)
        return camera.lookAt(width / 2, height / 2, 0)
      }}
      dpr={window.devicePixelRatio}
      frameLoop="demand"
    >
      {/*<svg width={width} height={height}>*/}
      {/*  <line x1={xStart} x2={xEnd} y1={yEnd} y2={yEnd} stroke="lack" />*/}
      {/*  <line x1={xStart} x2={xStart} y1={yEnd} y2={yStart} stroke="red" />*/}
      {/*  <g className="ticks">*/}
      {/*    {ticks.map((t, i) => {*/}
      {/*      const x = xScale(t)*/}
      {/*      return (*/}
      {/*        <React.Fragment key={i}>*/}
      {/*          <line x1={x} x2={x} y1={yEnd} y2={yEnd + 5} stroke="red" />*/}
      {/*          <text x={x} y={yEnd + 20} fill="red" textAnchor="middle" fontSize={10}>*/}
      {/*            {t}*/}
      {/*          </text>*/}
      {/*        </React.Fragment>*/}
      {/*      )*/}
      {/*    })}*/}
      {/*  </g>*/}
      {/*</svg>*/}
      {allSeries.map(series => (
        <Line key={series.id} series={series} />
      ))}
    </Canvas>
  )
})
