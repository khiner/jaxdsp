import ChartEventAccumulator from './ChartEventAccumulator'

export default class TrainTimeSeriesAccumulator extends ChartEventAccumulator {
  constructor() {
    super()
    this.data = [{ id: 'loss', label: 'Loss', data: [], permanent: true }]
  }

  doAccumulate(events = []) {
    events.forEach(({ time_ms, loss }) => this.data[0].data.push({ x: time_ms, y: loss }))
  }
}
