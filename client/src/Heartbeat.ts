// These types map directly to the server heartbeat JSON object.

import { Param } from './components/Processor'

export interface TraceEvent {
  function_name: string
  label: string
  start_time_ms: number
  end_time_ms: number
  duration_ms: number
}

export interface TrainStepEvent {
  time_ms: number
  processor_names: string[]
  params: Record<string, Param>
  loss: number
}

export type HeartbeatEvent = TraceEvent | TrainStepEvent

export default interface Heartbeat {
  train_events: TrainStepEvent[]
  trace_events: TraceEvent[]
}
