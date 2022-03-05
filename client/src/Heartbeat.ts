// These types map directly to the JSON in the server response.

export interface Param {
  name: string
  default_value: number
  min_value: number
  max_value: number
  log_scale?: boolean
}

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
