import React, { useEffect, useState } from 'react'
import ProcessorGraphBuilder from './ProcessorGraphBuilder'
import TrainTimeSeriesAccumulator from '../chart_event_accumulators/TrainTimeSeriesAccumulator'
import TraceFlameChartAccumulator from '../chart_event_accumulators/TraceFlameChartAccumulator'
import TraceTimeSeriesAccumulator from '../chart_event_accumulators/TraceTimeSeriesAccumulator'
import { last } from '../util/array'
import Monitor from './Monitor'
import { ProcessorType } from './Processor'
import type Heartbeat from '../Heartbeat'
import { Data } from './charts/Chart'

const trainTimeSeriesAccumulator = new TrainTimeSeriesAccumulator(true)
const traceTimeSeriesAccumulator = new TraceTimeSeriesAccumulator()
const traceFlameAccumulator = new TraceFlameChartAccumulator()

interface Props {
  clientUid: string
  processorDefinitions: any[]
  selectedProcessors: any[]
  setSelectedProcessors: (ps: ProcessorType[]) => void
  onError: (error: any) => void
}

export default function ({
  clientUid,
  processorDefinitions,
  selectedProcessors,
  setSelectedProcessors,
  onError,
}: Props) {
  const [estimatedParams, setEstimatedParams] = useState(undefined)
  const [trainTimeSeriesData, setTrainTimeSeriesData] = useState<Data | undefined>()
  const [traceTimeSeriesData, setTraceTimeSeriesData] = useState<Data | undefined>()
  const [traceFlameData, setTraceFlameData] = useState<Data | undefined>()

  useEffect(() => {
    if (clientUid === null) return

    // TODO wss? (SSL)
    const ws = new WebSocket('ws://127.0.0.1:8765/')
    ws.onopen = () => {
      ws.send(JSON.stringify({ client_uid: clientUid }))
    }
    ws.onmessage = message => {
      const heartbeat: Heartbeat = JSON.parse(message.data)
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

    return () => {
      trainTimeSeriesAccumulator.reset()
      traceTimeSeriesAccumulator.reset()
      traceFlameAccumulator.reset()
      ws.close()
    }
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
      <Monitor
        trainTimeSeriesData={trainTimeSeriesData}
        traceTimeSeriesData={traceTimeSeriesData}
        traceFlameData={traceFlameData}
      />
    </div>
  )
}
