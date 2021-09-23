import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'

import FatLine from '../primitives/FatLine'
import { scaleLinear } from 'd3-scale'
import { setPosition, VERTICES_PER_POSITION } from '../primitives/Rectangle'

export default React.memo(
  ({ series, dimensions, strokeWidth = 2, strokeColor = '#666', maxNumPoints = 10_000 }) => {
    const ref = useRef()
    const [positions, setPositions] = useState(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))

    useEffect(() => {
      setPositions(new Float32Array(VERTICES_PER_POSITION * maxNumPoints))
    }, [maxNumPoints])

    const { x, y, width, height } = dimensions

    useLayoutEffect(() => {
      const { data } = series
      if (!data?.length) return

      const { xDomain, yDomain } = series
      const xScale = scaleLinear().domain(xDomain).range([x, width])
      const yScale = scaleLinear().domain(yDomain).range([y, height])

      let i = 0
      data.forEach(({ x, y }) => {
        i = setPosition(positions, undefined, i, xScale(x), yScale(y))
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
