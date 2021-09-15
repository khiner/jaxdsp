import React from 'react'
import { Line } from '@nivo/line'
import { timeFormat } from 'd3-time-format'
import { last } from '../../util/array'

const DEFAULT_WINDOW_MILLIS = 8 * 1_000
const formatTime = timeFormat('%M:%S.%L')
const allSeries = []

// `value` is an object, whose keys will be used as labels,
// and whose values are arrays of two-dimensional [epochMillis, value] pairs, where value is a number or an object.
// These values will be tracked over time.
// Valid examples:
//    <RealTimeChart value={{ loss: 0.01 }} showKeys={['loss']} />
//    <RealTimeChart value={{ process_time: [0.01, 0.2], train_time: 0.21 }} hideKeys={['loss']} />
export default function RealTimeChart({ value, showKeys = [], hideKeys = [] }) {
  if (value !== undefined) {
    Object.entries(value).forEach(([key, value]) => {
      const series =
        allSeries.find(({ id }) => id === key) ||
        (allSeries.push({
          id: key,
          label: key,
          data: [],
        }) &&
          last(allSeries))
      series.data.push(
        ...value.map(([epochMillis, value]) => ({
          x: epochMillis,
          y: value,
        }))
      )
    })
  }

  const nowMillis = Date.now()
  allSeries.forEach(series => {
    series.data = series.data.filter(({ x }) => x >= nowMillis - DEFAULT_WINDOW_MILLIS)
  })

  if (allSeries.length === 0) return null

  const shownSeries = allSeries.filter(
    ({ id }) => !hideKeys.includes(id) && (showKeys.length === 0 || showKeys.includes(id))
  )
  const allValues = shownSeries.flatMap(({ data }) => data)
  const xs = allValues.map(({ x }) => x)
  const ys = allValues.map(({ y }) => y)

  return (
    <Line
      width={800}
      height={300}
      margin={{ top: 30, right: 50, bottom: 60, left: 50 }}
      data={shownSeries}
      xFormat={formatTime}
      xScale={{ type: 'linear', min: Math.min(...xs), max: Math.max(...xs) }}
      yScale={{ type: 'linear', min: 0, max: Math.max(...ys) * 1.1 }}
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
