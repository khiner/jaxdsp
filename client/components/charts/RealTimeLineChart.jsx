import React from 'react'
import { Line } from '@nivo/line'
import { timeFormat } from 'd3-time-format'
import { last } from '../../util/array'

const DEFAULT_WINDOW_MILLIS = 10 * 1_000
const formatMinutesSeconds = timeFormat('%M:%S')
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

  if (allSeries.length === 0) return null

  const nowMillis = Date.now()
  allSeries.forEach(series => {
    series.data = series.data.filter(({ x }) => x >= nowMillis - DEFAULT_WINDOW_MILLIS)
  })
  const shownSeries = allSeries.filter(
    ({ id }) => !hideKeys.includes(id) && (showKeys.length === 0 || showKeys.includes(id))
  )
  const maxY = Math.max(...shownSeries.flatMap(({ data }) => data).map(({ y }) => y))

  return (
    <Line
      width={800}
      height={300}
      margin={{ top: 30, right: 0, bottom: 50, left: 70 }}
      data={shownSeries}
      xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
      yScale={{ type: 'linear', min: 0, max: maxY * 1.1 }}
      axisLeft={{
        orient: 'left',
        legend: 'Execution duration (ms)',
        legendOffset: -60,
        legendPosition: 'middle',
        format: value => Number(value).toFixed(maxY < 0.01 ? 4 : maxY < 0.1 ? 3 : maxY < 1 ? 2 : 1),
      }}
      axisBottom={{ format: formatMinutesSeconds }}
      enablePoints={false}
      enableGridX={true}
      curve="monotoneX"
      animate={false}
      isInteractive={false}
      legends={[
        {
          anchor: 'bottom',
          direction: 'row',
          itemsSpacing: 0,
          translateY: 50,
          itemWidth: 120,
          itemHeight: 20,
          symbolSize: 10,
          symbolShape: 'circle',
        },
      ]}
      layers={['grid', 'lines', 'points', 'axes', 'legends']}
    />
  )
}
