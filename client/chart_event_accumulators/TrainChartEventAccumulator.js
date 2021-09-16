import ChartEventAccumulator from './ChartEventAccumulator'

export default class TrainChartEventAccumulator extends ChartEventAccumulator {
  constructor() {
    super()
    this.data = [{ id: 'loss', label: 'Loss', data: [] }]
  }

  doAccumulate(events = []) {
    events.forEach(({ time_ms, loss }) => this.data[0].data.push({ x: time_ms, y: loss }))
  }
}
