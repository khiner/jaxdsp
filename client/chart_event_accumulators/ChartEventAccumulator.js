import { last } from '../util/array'
import { getChartColor } from '../util/colors'
import { min, max, p25, median, p75 } from '../util/stats'

const DEFAULT_EXPIRATION_MILLIS = 10 * 1_000

const getMinTimeMillis = datum => {
  if (!datum) return 0
  const { x1, x } = datum
  return x1 || x
}

const getMaxTimeMillis = datum => {
  if (!datum) return 0
  const { x2, x } = datum
  return x2 || x
}

export default class ChartEventAccumulator {
  // If `summarize` is true, for each series, accumulate statistics for box plots into an additional `summaryData` field
  constructor(summarize = false) {
    this.data = { xDomain: [0, 0], yDomain: [0, 0], data: [] }
    this.allSeenSeriesIds = [] // Used to maintain color associations for series IDs that are removed and come back
    this.summarize = summarize
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
  push(seriesId, datum) {
    const series = this.findOrAddSeries(seriesId)
    if (getMinTimeMillis(last(series.data)) === getMinTimeMillis(datum)) series.data.pop()
    series.data.push(datum)

    if (this.summarize) {
      if (series.summaryData === undefined) series.summaryData = []
      const lastSummaryDatum = last(series.summaryData)
      let active
      if (lastSummaryDatum?.values !== undefined) {
        active = lastSummaryDatum
      } else {
        active = {
          values: [], // when full, this will be deleted
          x1: getMinTimeMillis(datum),
          x2: getMaxTimeMillis(datum),
          count: 0,
          min: 0.0,
          p25: 0.0,
          median: 0.0,
          p75: 0.0,
          max: 0.0,
        }
        series.summaryData.push(active)
      }

      const { values } = active
      values.push(datum.y)
      active.x2 = getMaxTimeMillis(datum)
      active.min = min(values)
      active.p25 = p25(values)
      active.median = median(values)
      active.p75 = p75(values)
      active.max = max(values)
      active.count = values.length

      if (values.length === 10) {
        active.numValues = values.length
        delete active.values
      }
    }
  }

  expireData(expirationDurationMillis) {
    const expirationMillis = Date.now() - expirationDurationMillis
    const allSeries = this.allSeries()
    allSeries.forEach(({ data }) => {
      // Assuming events come in time-order
      while (getMinTimeMillis(data[0]) && getMinTimeMillis(data[0]) < expirationMillis) {
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
        min(data.map(datum => getMinTimeMillis(datum))),
        max(data.map(datum => getMaxTimeMillis(datum))),
      ]
      const ys = data.map(({ y }) => y)
      series.yDomain = [min(ys), max(ys)]
    })
    // Cross-series domains
    this.data.xDomain = [
      min(allSeries.map(({ xDomain }) => xDomain[0])),
      max(allSeries.map(({ xDomain }) => xDomain[1])),
    ]
    this.data.yDomain = [
      min(allSeries.map(({ yDomain }) => yDomain[0])),
      max(allSeries.map(({ yDomain }) => yDomain[1])),
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
