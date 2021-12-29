// From https://stackoverflow.com/a/55297611/780425

export const asc = array => array.sort((a, b) => a - b) // ascending
export const sum = array => array.reduce((a, b) => a + b, 0)
export const min = array => Math.min(...array)
export const max = array => Math.max(...array)
export const mean = array => sum(array) / array.length

export const std = array => {
  const mu = mean(array)
  const diffArray = array.map(a => (a - mu) ** 2)
  return Math.sqrt(sum(diffArray) / (array.length - 1))
}

export const percentile = (array, p) => {
  const sorted = asc(array)
  const position = (sorted.length - 1) * p
  const positionFloor = Math.floor(position)
  const remainder = position - positionFloor
  return sorted[positionFloor + 1] !== undefined
    ? sorted[positionFloor] + remainder * (sorted[positionFloor + 1] - sorted[positionFloor])
    : sorted[positionFloor]
}

export const p25 = array => percentile(array, 0.25)
export const p50 = array => percentile(array, 0.5)
export const p75 = array => percentile(array, 0.75)
export const median = array => p50(array)
