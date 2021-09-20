import React from 'react'
import { extend } from '@react-three/fiber'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry'
import { Line2 } from 'three/examples/jsm/lines/Line2'

extend({ LineMaterial, LineGeometry, Line2 })

export default React.memo(
  React.forwardRef(({ width, height, strokeWidth = 2, strokeColor = '#666' }, ref) => (
    <line2>
      <lineGeometry attach="geometry" ref={ref} />
      <lineMaterial
        attach="material"
        color={strokeColor}
        linewidth={strokeWidth}
        resolution={[width, height]}
      />
    </line2>
  ))
)
