import React, { useState } from 'react'
import { scaleLinear } from 'd3-scale'

const percent = ratio => `${100 * ratio}%`

// Example:
//    <FlameChart data={
//      xDomain: [1631772930783, 1631772941650],
//      data: [
//        { id: 'test', label: 'Test', data: [{ start_time_ms: 1631772930783, end_time_ms: 1631772941650  }]}
//      ]
//    }/>
// Note that `duration_ms` could be different than end_time_ms - start_time_ms, since it's
// calculated using Python's more accurate `time.perf_counter`.
export default React.memo(({ data }) => {
  const [hoveringDatumId, setHoveringDatumId] = useState(undefined)

  if (!data) return null

  const { data: allSeries, xDomain } = data
  const numSeries = allSeries?.length
  if (!numSeries) return null

  const xScale = scaleLinear().domain(xDomain).range([0, 1])
  const height = 40 * numSeries

  return (
    <div style={{ display: 'flex', flexDirection: 'row' }}>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-around' }}>
        {allSeries.map(({ label }) => (
          <div key={label} style={{ padding: '0 1em', fontWeight: 'bold', textAlign: 'right' }}>
            {label}
          </div>
        ))}
      </div>
      <svg width="50%" height={height}>
        {allSeries.map(({ data, color }, seriesIndex) =>
          data.map(({ id, start_time_ms, end_time_ms }) => (
            <rect
              key={id}
              onMouseOver={() => setHoveringDatumId(id)}
              onMouseLeave={() => {
                if (hoveringDatumId === id) setHoveringDatumId(undefined)
              }}
              fill={hoveringDatumId === id ? '#00FF00' : color}
              x={percent(xScale(start_time_ms))}
              y={percent(seriesIndex / numSeries)}
              width={percent(Math.max(xScale(end_time_ms) - xScale(start_time_ms), 0.002))}
              height={percent(1.0 / numSeries)}
            />
          ))
        )}
      </svg>
    </div>
  )
})
