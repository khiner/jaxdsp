import ChartEventAccumulator from './ChartEventAccumulator'

const cumulativeSeriesWidth = series => series?.data?.reduce((total, { x1, x2 }) => total + (x2 - x1), 0)

export default class TraceFlameChartAccumulator extends ChartEventAccumulator {
  constructor(summarize = false) {
    super(summarize)
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
      // `x1` and `x2` may get overridden.
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
          // This should never happen, but not gonna freak out with an exception if it does.
          // Instead, trust the `duration_ms` value to generate the start time.
          x1 = x2 - duration_ms
          laneIndex = 0
        }
      }
      if (x1 === undefined) throw 'Flame chart series datum must have a start time'

      const seriesId = `${label}-${laneIndex}`
      this.push(seriesId, { id: `${seriesId}-${x1}`, x1, x2: x2 || now_ms, duration_ms }, label)
    })
    this.allSeries().sort((sA, s) => cumulativeSeriesWidth(sA) - cumulativeSeriesWidth(s))
  }
}
