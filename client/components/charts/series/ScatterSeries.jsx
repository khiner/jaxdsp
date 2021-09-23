import React, { useLayoutEffect, useRef } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'

export default React.memo(
  ({ series, dimensions, pointRadius = 3, pointColor = '#F66', maxNumPoints = 10_000 }) => {
    const ref = useRef()

    useLayoutEffect(() => {
      const { data } = series
      if (!data?.length) return

      const { x, y, width, height } = dimensions
      const { xDomain, yDomain } = series
      const xScale = scaleLinear()
        .domain(xDomain)
        .range([x, x + width])
      const yScale = scaleLinear()
        .domain(yDomain)
        .range([y, y + height])

      const mesh = ref.current
      const transform = new THREE.Matrix4()
      data.forEach(({ x, y }, i) => {
        transform.setPosition(xScale(x), yScale(y), 0)
        mesh.setMatrixAt(i, transform)
      })
      mesh.count = data.length - 1
      mesh.instanceMatrix.needsUpdate = true
    })

    return (
      <instancedMesh ref={ref} args={[null, null, maxNumPoints]}>
        <circleBufferGeometry args={[pointRadius]} />
        <meshBasicMaterial color={pointColor} />
      </instancedMesh>
    )
  }
)
