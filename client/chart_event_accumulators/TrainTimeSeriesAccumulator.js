import ChartEventAccumulator from './ChartEventAccumulator'

export default class TrainTimeSeriesAccumulator extends ChartEventAccumulator {
  constructor(summarize = false) {
    super(summarize)
    this.setAllSeries([{ id: 'loss', label: 'Loss', data: [], permanent: true }])
  }

  doAccumulate(events = []) {
    events.forEach(({ time_ms, loss }) => {
      this.push('loss', { x: time_ms, y: loss })
    })
  }
}
