import ChartEventAccumulator from './ChartEventAccumulator'

export default class TraceChartEventAccumulator extends ChartEventAccumulator {
  doAccumulate(events = []) {
    events.forEach(({ label, finish_time_ms, execution_duration_ms }) => {
      const series = this.findOrAddSeries(label)
      series.data.push({ x: finish_time_ms, y: execution_duration_ms })
    })
  }
}
