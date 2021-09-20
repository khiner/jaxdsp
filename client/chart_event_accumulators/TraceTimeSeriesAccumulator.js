import ChartEventAccumulator from './ChartEventAccumulator'

export default class TraceTimeSeriesAccumulator extends ChartEventAccumulator {
  doAccumulate(events = []) {
    events
      .filter(({ duration_ms }) => duration_ms)
      .forEach(({ label, end_time_ms, duration_ms }) => {
        this.push(label, end_time_ms, duration_ms)
      })
  }
}
