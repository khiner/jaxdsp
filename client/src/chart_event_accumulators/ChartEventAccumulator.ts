import { last } from '../util/array'
import { getChartColor } from '../util/colors'
import { max, median, min, p25, p75 } from '../util/stats'
import {
  Data,
  InnerSeries,
  SeriesData,
  SeriesDatum,
  SeriesSummaryData,
  SeriesSummaryDatum,
} from '../components/charts/Chart'
import type { HeartbeatEvent } from '../Heartbeat'

const DEFAULT_EXPIRATION_MILLIS = 5 * 1_000

const getMinTimeMillis = (datum: SeriesDatum | SeriesSummaryDatum) =>
  !datum ? 0 : 'x' in datum ? datum.x1 || datum.x : datum.x1

const getMaxTimeMillis = (datum: SeriesDatum | SeriesSummaryDatum) =>
  !datum ? 0 : 'x' in datum ? datum.x2 || datum.x : datum.x2

const expireSeriesData = (data: SeriesData | SeriesSummaryData, expirationDurationMillis) => {
  if (!data) return

  const expirationMillis = Date.now() - expirationDurationMillis
  // Assuming events come in time-ordered
  while (getMaxTimeMillis(data[0]) && getMaxTimeMillis(data[0]) < expirationMillis) {
    data.shift()
  }
}

export default class ChartEventAccumulator {
  data: Data
  allSeenSeriesIds: string[]
  summarize: boolean

  // If `summarize` is true, for each series, accumulate statistics for box plots into an additional `summaryData` field
  constructor(summarize = false) {
    this.summarize = summarize
    this.reset()
  }

  allSeries(): InnerSeries[] {
    return this.data.data
  }

  accumulate(events: HeartbeatEvent[] = [], expirationMillis = DEFAULT_EXPIRATION_MILLIS): Data {
    this.doAccumulate(events)
    this.expireData(expirationMillis)
    this.refreshDomains()
    return this.data
  }

  reset() {
    this.data = { xDomain: [0, 0], yDomain: [0, 0], data: [] }
    this.allSeenSeriesIds = [] // Used to maintain color associations for series IDs that are removed and come back
  }

  protected doAccumulate(events: HeartbeatEvent[] = []) {
    throw `Unimplemented abstract method \`doAccumulate\` called with ${events.length} events`
  }

  // Note: be sure to call `expireData` and `refreshDomain` after pushing all data!
  protected push(seriesId: string, datum: SeriesDatum, label?: string) {
    const series = this.findOrAddSeries(seriesId, label)
    if (getMinTimeMillis(last(series.data)) === getMinTimeMillis(datum)) series.data.pop()
    series.data.push(datum)

    if (this.summarize) {
      if (!series.summaryData) series.summaryData = []

      const { summaryData } = series
      if (!last(summaryData)?.values?.length) {
        summaryData.push({
          values: [], // when full, this will be deleted
          x1: getMinTimeMillis(datum),
          x2: getMaxTimeMillis(datum),
          count: 0,
          min: 0.0,
          p25: 0.0,
          median: 0.0,
          p75: 0.0,
          max: 0.0,
        })
      }

      const active = last(summaryData)
      const { values } = active
      values.push(datum.y)
      active.x2 = getMaxTimeMillis(datum)
      active.min = min(values)
      active.p25 = p25(values)
      active.median = median(values)
      active.p75 = p75(values)
      active.max = max(values)
      active.count = values.length

      if (values.length === 10) delete active.values
    }
  }

  private setAllSeries(allSeries) {
    this.data.data = allSeries
  }

  private expireData(expirationDurationMillis) {
    const allSeries = this.allSeries()
    allSeries.forEach(({ data, summaryData }) => {
      expireSeriesData(data, expirationDurationMillis)
      expireSeriesData(summaryData, expirationDurationMillis)
    })
    this.setAllSeries(allSeries.filter(({ data, permanent }) => permanent || data.length > 0))
  }

  private refreshDomains() {
    // Single-series domains
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

  private findOrAddSeries(id: string, label?: string): InnerSeries {
    if (!this.allSeenSeriesIds.includes(id)) this.allSeenSeriesIds.push(id)
    const color = getChartColor(this.allSeenSeriesIds.indexOf(id))
    const allSeries = this.allSeries()
    const matchingSeries = allSeries.find(({ id: seriesId }) => seriesId === id)
    if (matchingSeries) return matchingSeries

    allSeries.push({ id, label: label || `${id}`, color, data: [] })
    return last(allSeries)
  }
}
