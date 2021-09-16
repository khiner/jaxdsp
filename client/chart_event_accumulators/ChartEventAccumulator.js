import { last } from '../util/array'

const DEFAULT_EXPIRATION_MILLIS = 10 * 1_000
export default class ChartEventAccumulator {
  constructor() {
    this.data = []
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
      series.data = series.data.filter(({ x }) => x >= nowMillis - expirationMillis)
    })
  }

  findOrAddSeries(label) {
    return (
      this.data.find(({ id }) => id === label) ||
      (this.data.push({
        id: label,
        label,
        data: [],
      }) &&
        last(this.data))
    )
  }
}
