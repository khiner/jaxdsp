import React from 'react'
import { Vector3 } from 'three'

const ClipPlane = ({ axis, position, inverted = false }) => {
  const unit = inverted ? -1 : 1
  const normal = new Vector3(axis === 'x' ? unit : 0, axis === 'y' ? unit : 0, axis === 'z' ? unit : 0)
  return <plane attachArray="clippingPlanes" args={[normal, -unit * position]} />
}

// Clip planes defined in the following order: Top, Right, Bottom, Left
export default ({ dimensions }) => {
  const { x, y, width, height } = dimensions
  return (
    <>
      <ClipPlane axis="y" position={y + height} inverted />
      <ClipPlane axis="x" position={x + width} inverted />
      <ClipPlane axis="y" position={y} />
      <ClipPlane axis="x" position={x} />
    </>
  )
}
