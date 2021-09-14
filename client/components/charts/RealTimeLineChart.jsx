import React from 'react'
import { Line } from '@nivo/line'
import { timeFormat } from 'd3-time-format'
import { last } from '../../util/array'

const formatTime = timeFormat('%M:%S.%L')

const allSeries = []

const DEFAULT_WINDOW_MILLIS = 10 * 1000

const append = (value, values, windowMillis = DEFAULT_WINDOW_MILLIS) => {
  const nowMillis = new Date().getTime()
  values.push({ x: nowMillis, y: value })
  while (values[0].x < nowMillis - windowMillis) values.shift()
}

// `value` is an object, whose keys will be used as labels,
// and whose numeric (or array-of-numerics) value will be tracked over time
// Valid examples:
//    <RealTimeChart value={{ loss: 0.01 }} showKeys={['loss']} />
//    <RealTimeChart value={{ process_time: [0.01, 0.2], train_time: 0.21 }} hideKeys={['loss']} />
export default function RealTimeChart({ value, showKeys = [], hideKeys = [] }) {
  if (value !== undefined) {
    Object.entries(value).forEach(([key, numericValue]) => {
      const series =
        allSeries.find(({ id }) => id === key) ||
        (allSeries.push({
          id: key,
          label: key,
          data: [],
        }) &&
          last(allSeries))
      if (Array.isArray(numericValue)) {
        numericValue.forEach(v => append(v, series.data))
      } else {
        append(numericValue, series.data)
      }
    })
  }

  if (allSeries.length === 0) return null

  const shownSeries = allSeries.filter(
    ({ id }) => !hideKeys.includes(id) && (showKeys.length === 0 || showKeys.includes(id))
  )
  const allValues = shownSeries.flatMap(({ data }) => data)
  const minX = Math.min(...allValues.map(({ x }) => x))
  const maxX = Math.max(...allValues.map(({ x }) => x))
  const maxY = Math.max(...allValues.map(({ y }) => y))

  return (
    <Line
      width={800}
      height={300}
      margin={{ top: 30, right: 50, bottom: 60, left: 50 }}
      data={shownSeries}
      xFormat={formatTime}
      xScale={{ type: 'linear', min: minX, max: maxX }}
      yScale={{ type: 'linear', min: 0, max: maxY + maxY * 0.1 }}
      axisBottom={{ format: formatTime }}
      enablePoints={false}
      enableGridX={true}
      curve="monotoneX"
      animate={false}
      motionStiffness={120}
      motionDamping={50}
      isInteractive={false}
      enableSlices={false}
      useMesh={false}
      theme={{
        axis: { ticks: { text: { fontSize: 14 } } },
        grid: { line: { stroke: '#ddd', strokeDasharray: '1 2' } },
      }}
    />
  )
}
