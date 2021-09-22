import React, { useLayoutEffect, useRef } from 'react'
import * as THREE from 'three'
import { scaleLinear } from 'd3-scale'
import { useThree } from '@react-three/fiber'

export default React.memo(({ series, pointRadius = 3, pointColor = '#F66', maxNumPoints = 10_000 }) => {
  const ref = useRef()
  const { size } = useThree()
  const { width, height } = size

  useLayoutEffect(() => {
    const { data } = series
    if (!data?.length) return

    const { xDomain, yDomain } = series
    const xScale = scaleLinear().domain(xDomain).range([0, width])
    const yScale = scaleLinear().domain(yDomain).range([0, height])

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
})
