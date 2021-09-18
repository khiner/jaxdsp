import React, { useState } from 'react'

const percent = ratio => `${100 * ratio}%`

// Example:
//    <FlameChart data={[
//        { id: 'test', label: 'Test', data: [{ start_time_ms: 1631772930783, end_time_ms: 1631772941650  }]}
//      ]}
//    />
// Note that `duration_ms` could be different than end_time_ms - start_time_ms, since it's
// calculated using Python's more accurate `time.perf_counter`.
function FlameChart({ data }) {
  if (!data?.length) return null

  const [hoveringDatumId, setHoveringDatumId] = useState(undefined)

  const allPoints = data.flatMap(({ data }) => data)
  if (allPoints.length === 0) return null

  const minTimeMs = Math.min(...allPoints.map(({ start_time_ms }) => start_time_ms))
  const maxTimeMs = Math.max(...allPoints.map(({ end_time_ms }) => end_time_ms))
  const timeRangeMs = maxTimeMs - minTimeMs
  const height = 40 * data.length

  return (
    <div style={{ display: 'flex', flexDirection: 'row' }}>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-around' }}>
        {data.map(({ label }) => (
          <div key={label} style={{ padding: '0 1em', fontWeight: 'bold', textAlign: 'right' }}>
            {label}
          </div>
        ))}
      </div>
      <svg width="50%" height={height}>
        {data.map((series, seriesIndex) =>
          series.data.map(({ id, start_time_ms, end_time_ms }) => (
            <rect
              key={id}
              onMouseOver={() => setHoveringDatumId(id)}
              onMouseLeave={() => {
                if (hoveringDatumId === id) setHoveringDatumId(undefined)
              }}
              fill={hoveringDatumId === id ? '#00FF00' : series.color}
              x={percent((start_time_ms - minTimeMs) / timeRangeMs)}
              y={percent(seriesIndex / data.length)}
              width={percent(Math.max((end_time_ms - start_time_ms) / timeRangeMs, 0.002))}
              height={percent(1.0 / data.length)}
            />
          ))
        )}
      </svg>
    </div>
  )
}

export default React.memo(FlameChart)
