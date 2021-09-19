import React from 'react'
import { Canvas } from '@react-three/fiber'
import Line from './webgl/Line'
import { timeFormat } from 'd3-time-format'

const formatMinutesSeconds = timeFormat('%M:%S')

// `data` is a list of with `x` values assumed to be milliseconds since epoch.
// Example:
//    <TimeSeriesChart data={[{ id: 'test', label: 'Test', data: [{ x: 1631772930783, y: 0.01 }]}]} />
// TODO show points for start/end of contiguous ranges
export default React.memo(({ data, width = 400, height = 200 }) => {
  if (!data?.length) return null

  const allPoints = data.flatMap(({ data }) => data)
  if (allPoints.length === 0) return null

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
      {data.map(series => (
        <Line key={series.id} data={series.data} />
      ))}
    </Canvas>
  )
})
