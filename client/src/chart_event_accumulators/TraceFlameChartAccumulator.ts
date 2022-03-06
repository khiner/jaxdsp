import ChartEventAccumulator from './ChartEventAccumulator'
import type { TraceEvent } from '../Heartbeat'

const cumulativeSeriesWidth = series => series?.data?.reduce((total, { x1, x2 }) => total + (x2 - x1), 0)

export default class TraceFlameChartAccumulator extends ChartEventAccumulator {
  startEventsForLabel: Record<string, TraceEvent[]> = {}

  constructor(summarize = false) {
    super(summarize)
  }

  // Each series is a "lane" corresponding to a function.
  // If a method with the same label is called more than once before closing (e.g. a recursive call),
  // a new series (lane) with a `${label}-${numActiveWithLabel}` key will be created.
  // Series are ordered by cumulative duration.
  doAccumulate(events: TraceEvent[] = []) {
    const nowMillis = Date.now()

    events.forEach(event => {
      const { label, duration_ms } = event
      let { start_time_ms: x1, end_time_ms: x2 } = event
      let laneIndex = 0

      if (x1 !== undefined) {
        if (!(label in this.startEventsForLabel)) this.startEventsForLabel[label] = []
        this.startEventsForLabel[label].push(event)
        laneIndex = this.startEventsForLabel[label].length - 1
      } else if (x2 !== undefined) {
        const startEvents = this.startEventsForLabel[label]
        if (startEvents?.length) {
          laneIndex = startEvents.length - 1
          x1 = startEvents.pop().start_time_ms
          if (startEvents.length === 0) delete this.startEventsForLabel[label]
        } else {
          // No corresponding start event for this end event. Something went wrong.
          // This should never happen, but if it does, just use `duration_ms` to derive the start time.
          // Note that `duration_ms` could be different from x1 - x2, since it's calculated using Python's
          // more accurate `time.perf_counter`.
          x1 = x2 - duration_ms
          laneIndex = 0
        }
      }
      if (x1 === undefined) throw 'Flame chart series datum must have a start time'

      const seriesId = `${label}-${laneIndex}`
      this.push(seriesId, { x1, x2: x2 || nowMillis }, label)
    })

    this.allSeries.sort((sA, s) => cumulativeSeriesWidth(sA) - cumulativeSeriesWidth(s))
  }
}
