import React, { useLayoutEffect, useRef } from 'react'
import * as THREE from 'three'

const MAX_NUM_POINTS = 100_000

const PointMesh = React.memo(
  React.forwardRef(({ pointRadius = 3, pointColor }, ref) => (
    <instancedMesh ref={ref} args={[null, null, MAX_NUM_POINTS]}>
      <circleBufferGeometry args={[pointRadius]} />
      <meshBasicMaterial color={pointColor} />
    </instancedMesh>
  ))
)

export default React.memo(
  ({ data, pointRadius = 2, pointColor = '#F66' }) => {
    const ref = useRef()
    useLayoutEffect(() => {
      const mesh = ref.current
      const transform = new THREE.Matrix4()
      data.forEach((d, i) => {
        transform.setPosition(d.x, d.y, 0)
        mesh.setMatrixAt(i, transform)
      })
      mesh.count = data.length - 1
      mesh.instanceMatrix.needsUpdate = true
    }, [data])

    return <PointMesh ref={ref} pointRadius={pointRadius} pointColor={pointColor} />
  },
  (prev, next) => {
    return prev.pointRadius === next.pointRadius && prev.pointColor === next.pointColor
  }
)
