import React, { useLayoutEffect, useRef } from 'react'
import { Matrix4 } from 'three'
import { scaleLinear } from 'd3-scale'
import ClipArea from '../ClipArea'
import type { InstancedMeshProps } from '@react-three/fiber'
import type { SeriesProps } from '../Chart'

interface Props extends SeriesProps {
  pointRadius?: number
}

export default React.memo(
  ({ series, dimensions, xDomain, yDomain, renderOrder = 0, pointRadius = 2 }: Props) => {
    const ref = useRef<InstancedMeshProps>()

    useLayoutEffect(() => {
      const { data } = series
      if (!data?.length) return

      const { x, y, width, height } = dimensions
      const xScale = scaleLinear()
        .domain(xDomain)
        .range([x, x + width])
      const yScale = scaleLinear()
        .domain(yDomain)
        .range([y, y + height])

      const mesh = ref.current
      const transform = new Matrix4()
      data.forEach(({ x, y }, i) => {
        transform.setPosition(xScale(x), yScale(y), 0)
        mesh.setMatrixAt(i, transform)
      })
      mesh.count = data.length - 1
      mesh.instanceMatrix.needsUpdate = true
    })

    // Should be able to do something like this to avoid refs, but it lags behind for me:
    // <Instances limit={1_000}>
    //   <circleBufferGeometry args={[pointRadius]} />
    //   <meshBasicMaterial color={fill} />
    //   {data.map(({ x, y }) => (
    //     <Instance position={[xScale(x), yScale(y), 0]} />
    //   ))}
    // </Instances>
    return (
      <instancedMesh ref={ref} args={[null, null, 1_000]} renderOrder={renderOrder}>
        <circleBufferGeometry args={[pointRadius]} />
        <meshBasicMaterial color={series.color} transparent={true} opacity={0.8}>
          <ClipArea dimensions={dimensions} />
        </meshBasicMaterial>
      </instancedMesh>
    )
  }
)
