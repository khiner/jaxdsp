import ChartEventAccumulator from './ChartEventAccumulator'

export default class TrainTimeSeriesAccumulator extends ChartEventAccumulator {
  doAccumulate(events = []) {
    events.forEach(({ time_ms, loss }) => this.push('loss', { x: time_ms, y: loss }, 'Loss'))
  }
}
