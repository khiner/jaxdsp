import { Color } from 'three'

const mapValuesRecursive = (object, f) =>
  Object.fromEntries(
    Object.entries(object).map(([k, v]) => [k, v === Object(v) ? mapValuesRecursive(v, f) : f(v)])
  )

const toThreeColors = colors => mapValuesRecursive(colors, color => new Color(color))

export default toThreeColors({
  background: 'white',
  border: 'black',
  axis: {
    text: '#333',
    stroke: '#333',
  },
  grid: {
    stroke: '#888',
  },
  series: {
    box: {
      fill: '#777',
      medianStroke: '#111',
      minMaxStroke: '#111',
      whiskerStroke: '#222',
    },
  },
})
