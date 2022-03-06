import React, { Children, ReactChildren, ReactElement, useEffect } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import type { Dimensions } from './Chart'

const DEFAULT_CHART_HEIGHT = 200

function AdaptiveCamera() {
  const {
    camera,
    size: { width, height },
  } = useThree()
  useEffect(() => {
    // Calculate camera z so that the top and bottom are exactly at the edges of the fov
    // Based on https://stackoverflow.com/a/13351534/780425
    // Adding height for extra space to not clip horizontal lines exactly at 0/height in half.
    const maxLineWidth = 4
    const z = (height + maxLineWidth) / (2 * Math.tan((camera.fov / 360) * Math.PI))
    camera.position.set(width / 2, height / 2, z)
    camera.lookAt(width / 2, height / 2, 0)
    camera.updateProjectionMatrix()
  }, [width, height])
  return null
}

interface Props {
  width?: number
  children?: ReactElement[]
}

/**
 All explicit child `dimensions` fields except `y` (x/width/height) will be respected, and not modified.
 Child `y` positions will be calculated assuming children are provided in top-to-bottom order, and should be
 stacked directly on top of each other.
 Any missing `dimensions` fields will be automatically filled, using the following rules:
   - Children receive a `width` of `ChartContext::width` (fill the `ChartContext` parent).
   - Children will be provided a default `x` of `0`, and a default `height` of `200px`.
*/
export default function ChartContext({ width = 400, children }: Props) {
  // Learned the hard way that `Children.map`, unlike `Array.map`, removes `undefined` values.
  children = Children.map(children, child => (React.isValidElement(child) ? child : undefined))
  const childrenDimensions: Dimensions[] = Children.map(children, child => {
    const { x, width: chartWidth, height: chartHeight } = child.props.dimensions || {}
    return { x: x || 0, width: chartWidth || width, height: chartHeight || DEFAULT_CHART_HEIGHT }
  })
  let y = 0
  for (let i = childrenDimensions.length - 1; i >= 0; i -= 1) {
    const childDimensions = childrenDimensions[i]
    childDimensions.y = y
    y += childDimensions.height
  }

  const height = y

  return (
    <Canvas
      style={{ width, height }}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true
      }}
      dpr={window.devicePixelRatio}
      frameloop="demand"
    >
      <AdaptiveCamera />
      {Children.map(children, (child, i) => React.cloneElement(child, { dimensions: childrenDimensions[i] }))}
    </Canvas>
  )
}
