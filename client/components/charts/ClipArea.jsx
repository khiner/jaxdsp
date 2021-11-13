import React from 'react'
import { Vector3 } from 'three'

const ClipPlane = ({ axis, distance, inverted = false }) => {
  const unit = inverted ? -1 : 1
  const normal = new Vector3(axis === 'x' ? unit : 0, axis === 'y' ? unit : 0, axis === 'z' ? unit : 0)
  return <plane attachArray="clippingPlanes" args={[normal, distance]} />
}

// Clip planes defined in the following order: Top, Right, Bottom, Left
export default ({ dimensions }) => {
  const { x, y, width, height } = dimensions
  return (
    <>
      <ClipPlane axis="y" distance={y + height} inverted />
      <ClipPlane axis="x" distance={x + width} inverted />
      <ClipPlane axis="y" distance={y} />
      <ClipPlane axis="x" distance={-x} />
    </>
  )
}
