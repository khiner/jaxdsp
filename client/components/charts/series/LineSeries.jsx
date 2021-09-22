import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'

import FatLine from '../primitives/FatLine'
import { scaleLinear } from 'd3-scale'

export default React.memo(
  ({ series, dimensions, strokeWidth = 2, strokeColor = '#666', maxNumPoints = 10_000 }) => {
    const ref = useRef()
    const [positions, setPositions] = useState(new Float32Array(3 * maxNumPoints))

    useEffect(() => {
      setPositions(new Float32Array(3 * maxNumPoints))
    }, [maxNumPoints])

    const { x, y, width, height } = dimensions

    useLayoutEffect(() => {
      const { data } = series
      if (!data?.length) return

      const { xDomain, yDomain } = series
      const xScale = scaleLinear().domain(xDomain).range([x, width])
      const yScale = scaleLinear().domain(yDomain).range([y, height])

      data.forEach(({ x, y }, i) => {
        positions[i * 3] = xScale(x)
        positions[i * 3 + 1] = yScale(y)
        positions[i * 3 + 2] = 0
      })

      const geometry = ref.current
      geometry.setPositions(positions)
      geometry.instanceCount = data.length - 1
    })

    return (
      <FatLine ref={ref} width={width} height={height} strokeWidth={strokeWidth} strokeColor={strokeColor} />
    )
  }
)
