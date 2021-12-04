import React from 'react'
import { Canvas } from '@react-three/fiber'
import LineSeries from './series/LineSeries'
import BoxSeries from './series/BoxSeries'
import ScatterSeries from './series/ScatterSeries'
import Axis from './axes/Axis'
import { Line } from '@react-three/drei'
import colors from './colors'

// const formatMinutesSeconds = timeFormat('%M:%S')

// `data` is a list of with `x` values assumed to be milliseconds since epoch.
// Example:
//    <TimeSeriesChart data={{
//      xDomain: [1631772930783, 1631772930783],
//      yDomain: [0.01, 0.01],
//      data: [{ id: 'test', label: 'Test', data: [{ x: 1631772930783, y: 0.01 }]}]
//    }/>
// TODO show points for start/end of contiguous ranges
export default React.memo(({ data, width = 400, height = 200 }) => {
  const axisWidth = 80
  const seriesDimensions = { x: axisWidth, y: 0, width: width - axisWidth, height }

  if (!data) return null
  const { data: allSeries } = data
  if (!allSeries?.length) return null

  const { xDomain, yDomain } = data

  const Border = () => (
    <Line
      points={[
        [axisWidth, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [axisWidth, height, 0],
        [axisWidth, 0, 0],
      ]}
      color={colors.border}
      lineWidth={1}
    />
  )

  return (
    <Canvas
      style={{ width, height }}
      onCreated={({ camera, gl }) => {
        gl.localClippingEnabled = true
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
      {allSeries.map(series => (
        <LineSeries key={series.id} series={series} dimensions={seriesDimensions} />
      ))}
      {allSeries.map(series => (
        <BoxSeries key={series.id} series={series} dimensions={seriesDimensions} />
      ))}
      {allSeries.map(series => (
        <ScatterSeries key={series.id} series={series} dimensions={seriesDimensions} />
      ))}
      <Border />
      <Axis
        side="y"
        xDomain={xDomain}
        yDomain={yDomain}
        dimensions={{ x: 0, y: 0, width: axisWidth, height }}
      />
    </Canvas>
  )
})
