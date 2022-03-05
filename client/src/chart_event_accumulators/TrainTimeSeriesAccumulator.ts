import ChartEventAccumulator from './ChartEventAccumulator'
import type { TrainStepEvent } from '../Heartbeat'

export default class TrainTimeSeriesAccumulator extends ChartEventAccumulator {
  doAccumulate(events: TrainStepEvent[] = []) {
    events.forEach(({ time_ms, loss }) => this.push('loss', { x: time_ms, y: loss }, 'Loss'))
  }
}
