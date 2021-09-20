import ChartEventAccumulator from './ChartEventAccumulator'

export default class TrainTimeSeriesAccumulator extends ChartEventAccumulator {
  constructor() {
    super()
    this.setAllSeries([{ id: 'loss', label: 'Loss', data: [], permanent: true }])
  }

  doAccumulate(events = []) {
    events.forEach(({ time_ms, loss }) => {
      this.push('loss', time_ms, loss)
    })
  }
}
