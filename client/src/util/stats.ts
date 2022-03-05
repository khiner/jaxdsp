// From https://stackoverflow.com/a/55297611/780425

export const asc = (array: number[]) => array.sort((a, b) => a - b) // ascending
export const sum = (array: number[]) => array.reduce((a, b) => a + b, 0)
export const min = (array: number[]) => Math.min(...array)
export const max = (array: number[]) => Math.max(...array)
export const mean = (array: number[]) => sum(array) / array.length

export const std = (array: number[]) => {
  const mu = mean(array)
  const diffArray = array.map(a => (a - mu) ** 2)
  return Math.sqrt(sum(diffArray) / (array.length - 1))
}

export const percentile = (array: number[], p: number) => {
  const sorted = asc(array)
  const position = (sorted.length - 1) * p
  const positionFloor = Math.floor(position)
  const remainder = position - positionFloor
  return sorted[positionFloor + 1] !== undefined
    ? sorted[positionFloor] + remainder * (sorted[positionFloor + 1] - sorted[positionFloor])
    : sorted[positionFloor]
}

export const p25 = (array: number[]) => percentile(array, 0.25)
export const p50 = (array: number[]) => percentile(array, 0.5)
export const p75 = (array: number[]) => percentile(array, 0.75)
export const median = (array: number[]) => p50(array)
