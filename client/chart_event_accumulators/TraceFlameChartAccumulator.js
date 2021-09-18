import ChartEventAccumulator from './ChartEventAccumulator'
import { last } from '../util/array'

const cumulativeSeriesDuration = series =>
  series?.data?.reduce((total, { start_time_ms, end_time_ms }) => total + (end_time_ms - start_time_ms), 0)

export default class TraceFlameChartAccumulator extends ChartEventAccumulator {
  constructor() {
    super()
    this.startEventsForLabel = {} // chronological stack for each label
  }

  // Each series is a "lane" corresponding to a function.
  // If a method with the same label is called more than once before closing (e.g. a recursive call),
  // a new series (lane) with a `${label}-${numActiveWithLabel}` key will be created.
  // Series are ordered by cumulative duration.
  doAccumulate(events = []) {
    const now_ms = Date.now()

    events.forEach(event => {
      const { label, duration_ms } = event
      // `start_time_ms`, `end_time_ms` and `id` may get overridden.
      let { start_time_ms, end_time_ms } = event
      let laneIndex = 0

      if (start_time_ms !== undefined) {
        if (!(label in this.startEventsForLabel)) this.startEventsForLabel[label] = []
        this.startEventsForLabel[label].push(event)
        laneIndex = this.startEventsForLabel[label].length - 1
      } else if (end_time_ms !== undefined) {
        const startEvents = this.startEventsForLabel[label]
        if (startEvents?.length) {
          laneIndex = startEvents.length - 1
          start_time_ms = startEvents.pop().start_time_ms
          if (startEvents.length === 0) delete this.startEventsForLabel[label]
        } else {
          // No corresponding start event for this end event. Something went wrong.
          // This should never happen, but not gonna freak out with an exception if it does.
          // Instead, trust the `duration_ms` value to generate the start time.
          start_time_ms = end_time_ms - duration_ms
          laneIndex = 0
        }
      }
      if (start_time_ms === undefined) throw 'Flame chart series datum must have a start time'

      if (end_time_ms === undefined) end_time_ms = now_ms
      const series = this.findOrAddSeries(`${label}-${laneIndex}`, label)
      if (last(series.data)?.start_time_ms === start_time_ms) series.data.pop()

      series.data.push({ id: `${series.id}-${start_time_ms}`, start_time_ms, end_time_ms, duration_ms })
      this.data.sort(
        (seriesA, seriesB) => cumulativeSeriesDuration(seriesA) - cumulativeSeriesDuration(seriesB)
      )
    })
  }
}
