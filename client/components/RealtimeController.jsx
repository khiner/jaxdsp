import React, { useEffect, useState } from 'react'
import ProcessorGraphBuilder from './ProcessorGraphBuilder'
import TrainTimeSeriesAccumulator from '../chart_event_accumulators/TrainTimeSeriesAccumulator'
import { last } from '../util/array'
import FlameChart from './charts/FlameChart'
import TraceFlameChartAccumulator from '../chart_event_accumulators/TraceFlameChartAccumulator'
import TimeSeriesChart from './charts/TimeSeriesChart'
import TraceTimeSeriesAccumulator from '../chart_event_accumulators/TraceTimeSeriesAccumulator'

const trainTimeSeriesAccumulator = new TrainTimeSeriesAccumulator(true)
const traceTimeSeriesAccumulator = new TraceTimeSeriesAccumulator()
const traceFlameAccumulator = new TraceFlameChartAccumulator()

export default function ({
  clientUid,
  processorDefinitions,
  selectedProcessors,
  setSelectedProcessors,
  onError,
}) {
  const [estimatedParams, setEstimatedParams] = useState(null)
  const [trainTimeSeriesData, setTrainTimeSeriesData] = useState({})
  const [traceTimeSeriesData, setTraceTimeSeriesData] = useState({})
  const [traceFlameData, setTraceFlameData] = useState({})

  useEffect(() => {
    if (clientUid === null) return

    // TODO wss? (SSL)
    const ws = new WebSocket('ws://127.0.0.1:8765/')
    ws.onopen = () => {
      ws.send(JSON.stringify({ client_uid: clientUid }))
    }
    ws.onmessage = event => {
      const heartbeat = JSON.parse(event.data)
      const { train_events, trace_events } = heartbeat

      setTrainTimeSeriesData({ ...trainTimeSeriesAccumulator.accumulate(train_events) })
      setTraceTimeSeriesData({ ...traceTimeSeriesAccumulator.accumulate(trace_events) })
      setTraceFlameData({ ...traceFlameAccumulator.accumulate(trace_events) })

      const lastEstimatedParams = last(train_events)?.params
      if (lastEstimatedParams) setEstimatedParams(lastEstimatedParams)
    }
    ws.onclose = event => {
      const { wasClean, code } = event
      if (!wasClean) {
        onError(`WebSocket unexpectedly closed with code ${code}`)
      }
    }
    ws.onerror = () => {
      onError('WebSocket connection error')
    }
    return () => ws.close()
  }, [clientUid])

  return (
    <div>
      {processorDefinitions && (
        <ProcessorGraphBuilder
          processorDefinitions={processorDefinitions}
          selectedProcessors={selectedProcessors}
          estimatedParams={estimatedParams}
          onChange={setSelectedProcessors}
        />
      )}
      {trainTimeSeriesData && <TimeSeriesChart data={trainTimeSeriesData} />}
      {traceTimeSeriesData && <TimeSeriesChart data={traceTimeSeriesData} />}
      {traceFlameData && <FlameChart data={traceFlameData} />}
    </div>
  )
}
