import { last } from '../util/array'
import { getChartColor } from '../util/colors'

const DEFAULT_EXPIRATION_MILLIS = 10 * 1_000

export default class ChartEventAccumulator {
  constructor() {
    this.data = { xDomain: [0, 0], yDomain: [0, 0], data: [] }
    this.allSeenSeriesIds = [] // Used to maintain color associations for series IDs that are removed and come back
  }

  allSeries() {
    return this.data.data
  }

  setAllSeries(allSeries) {
    this.data.data = allSeries
  }

  accumulate(events = [], expirationMillis = DEFAULT_EXPIRATION_MILLIS) {
    this.doAccumulate(events)
    this.expireData(expirationMillis)
    this.refreshDomains()
    return this.data
  }

  doAccumulate(events = []) {
    throw `Unimplemented abstract method \`doAccumulate\` called with ${events.length} events`
  }

  // Note: be sure to call `expireData` and `refreshDomain` after pushing all data!
  push(seriesId, x, y) {
    const series = this.findOrAddSeries(seriesId)
    series.data.push({ x, y })
  }

  getMinTimeMillis(datum) {
    if (!datum) return 0
    const { start_time_ms, x } = datum
    return start_time_ms || x
  }

  getMaxTimeMillis(datum) {
    if (!datum) return 0
    const { end_time_ms, x } = datum
    return end_time_ms || x
  }

  expireData(expirationDurationMillis) {
    const expirationMillis = Date.now() - expirationDurationMillis
    const allSeries = this.allSeries()
    allSeries.forEach(({ data }) => {
      // Assuming events come in time-order
      while (this.getMinTimeMillis(data[0]) && this.getMinTimeMillis(data[0]) < expirationMillis) {
        data.shift()
      }
    })
    this.setAllSeries(allSeries.filter(({ data, permanent }) => permanent || data.length > 0))
  }

  refreshDomains() {
    // Single series domains
    const allSeries = this.allSeries()
    allSeries.forEach(series => {
      const { data } = series
      series.xDomain = [
        Math.min(...data.map(datum => this.getMinTimeMillis(datum))),
        Math.max(...data.map(datum => this.getMaxTimeMillis(datum))),
      ]
      const ys = data.map(({ y }) => y)
      series.yDomain = [Math.min(...ys), Math.max(...ys)]
    })
    // Cross-series domains
    this.data.xDomain = [
      Math.min(...allSeries.map(({ xDomain }) => xDomain[0])),
      Math.max(...allSeries.map(({ xDomain }) => xDomain[1])),
    ]
    this.data.yDomain = [
      Math.min(...allSeries.map(({ yDomain }) => yDomain[0])),
      Math.max(...allSeries.map(({ yDomain }) => yDomain[1])),
    ]
  }

  findOrAddSeries(id, label = undefined) {
    if (!this.allSeenSeriesIds.includes(id)) this.allSeenSeriesIds.push(id)
    const color = getChartColor(this.allSeenSeriesIds.indexOf(id))

    const allSeries = this.allSeries()
    return (
      allSeries.find(({ id: seriesId }) => seriesId === id) ||
      (allSeries.push({
        id,
        label: label || id,
        color,
        data: [],
      }) &&
        last(allSeries))
    )
  }
}
