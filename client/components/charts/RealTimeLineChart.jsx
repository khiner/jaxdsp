import React, { useEffect, useRef, useState } from 'react'
import { Line } from '@nivo/line'
import { timeFormat } from 'd3-time-format'
import { last } from '../../util/array'

const formatTime = timeFormat('%M:%S.%L')

const loss = []
let maxLossValue = 0.0

export default function RealTimeChart({ lossValue }) {
  if (lossValue !== undefined) {
    while (loss.length >= 100) loss.shift()
    loss.push({ x: new Date().getTime(), y: lossValue })
  }

  if (loss.length === 0) return null

  maxLossValue = Math.max(maxLossValue, lossValue)

  return (
    <Line
      width={900}
      height={400}
      margin={{ top: 30, right: 50, bottom: 60, left: 50 }}
      data={[{ id: 'loss', data: loss }]}
      xFormat={formatTime}
      xScale={{ type: 'linear', min: loss[0]?.x, max: last(loss)?.x }}
      yScale={{ type: 'linear', min: 0, max: maxLossValue + 0.1 }}
      axisBottom={{ format: formatTime }}
      enablePoints={false}
      enableGridX={true}
      curve="monotoneX"
      animate={false}
      motionStiffness={120}
      motionDamping={50}
      isInteractive={true}
      enableSlices={false}
      useMesh={true}
      theme={{
        axis: { ticks: { text: { fontSize: 14 } } },
        grid: { line: { stroke: '#ddd', strokeDasharray: '1 2' } },
      }}
    />
  )
}
