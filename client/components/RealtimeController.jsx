import React, { useEffect, useState } from 'react'
import ProcessorGraphBuilder from './ProcessorGraphBuilder'
import TimeSeriesChart from './charts/TimeSeriesChart'
import TrainChartEventAccumulator from '../chart_event_accumulators/TrainChartEventAccumulator'
import TraceChartEventAccumulator from '../chart_event_accumulators/TraceChartEventAccumulator'
import { last } from '../util/array'

const trainChartEventAccumulator = new TrainChartEventAccumulator()
const traceChartEventAccumulator = new TraceChartEventAccumulator()

export default function ({
  clientUid,
  processorDefinitions,
  selectedProcessors,
  setSelectedProcessors,
  setAudioStreamErrorMessage,
}) {
  const [estimatedParams, setEstimatedParams] = useState(null)
  const [trainChartData, setTrainChartData] = useState([])
  const [traceChartData, setTraceChartData] = useState([])

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

      setTraceChartData([...traceChartEventAccumulator.accumulate(trace_events)])
      setTrainChartData([...trainChartEventAccumulator.accumulate(train_events)])
      const lastEstimatedParams = last(train_events)?.params
      if (lastEstimatedParams) setEstimatedParams(lastEstimatedParams)
    }
    ws.onclose = event => {
      const { wasClean, code } = event
      if (!wasClean) {
        setAudioStreamErrorMessage(`WebSocket unexpectedly closed with code ${code}`)
      }
    }
    ws.onerror = () => {
      setAudioStreamErrorMessage('WebSocket connection error')
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
      {traceChartData && <TimeSeriesChart data={traceChartData} />}
      {trainChartData && <TimeSeriesChart data={trainChartData} />}
    </div>
  )
}
