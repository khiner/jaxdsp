import React from 'react'
import { Line } from '@nivo/line'
import { timeFormat } from 'd3-time-format'

const formatMinutesSeconds = timeFormat('%M:%S')

// `data` is a list of objects as expected by [nivo line charts](https://nivo.rocks/line/),
// with `x` values assumed to be milliseconds since epoch.
// Example:
//    <TimeSeriesChart data={[{ id: 0.01, label: 'Test', data: [{ x: 1631772930783, y: loss: 0.01 }]}]} />
// TODO show points for start/end of contiguous ranges: https://nivo.rocks/storybook/?path=/story/line--custom-line-style
function TimeSeriesChart({ data }) {
  if (!data?.length) return null

  const allPoints = data.flatMap(({ data }) => data)
  if (allPoints.length === 0) return null

  const maxY = Math.max(...allPoints.map(({ y }) => y))

  return (
    <Line
      width={800}
      height={300}
      margin={{ top: 30, right: 0, bottom: 50, left: 70 }}
      data={data}
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

export default React.memo(TimeSeriesChart)
