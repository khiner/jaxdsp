import { last } from '../util/array'
import { getChartColor } from '../util/colors'

const DEFAULT_EXPIRATION_MILLIS = 2 * 1_000
export default class ChartEventAccumulator {
  constructor() {
    this.data = []
    this.allSeenSeriesIds = [] // Used to maintain color associations for series IDs that are removed and come back
  }

  accumulate(events = [], expirationMillis = DEFAULT_EXPIRATION_MILLIS) {
    this.doAccumulate(events)
    this.expireData(expirationMillis)
    return this.data
  }

  doAccumulate(events = []) {
    throw `Unimplemented abstract method \`doAccumulate\` called with ${events.length} events`
  }

  expireData(expirationMillis) {
    const nowMillis = Date.now()
    this.data.forEach(series => {
      series.data = series.data.filter(
        ({ x, start_time_ms, end_time_ms }) =>
          (x || end_time_ms || start_time_ms) >= nowMillis - expirationMillis
      )
    })
    this.data = this.data.filter(({ data, permanent }) => permanent || data.length > 0)
  }

  findOrAddSeries(id, label = undefined) {
    if (!this.allSeenSeriesIds.includes(id)) this.allSeenSeriesIds.push(id)
    const color = getChartColor(this.allSeenSeriesIds.indexOf(id))

    return (
      this.data.find(({ id: seriesId }) => seriesId === id) ||
      (this.data.push({
        id,
        label: label || id,
        color,
        data: [],
      }) &&
        last(this.data))
    )
  }
}
