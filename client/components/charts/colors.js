import { Color } from 'three'

const mapValuesRecursive = (obj, func) =>
  Object.fromEntries(
    Object.entries(obj).map(([k, v]) => [k, v === Object(v) ? mapValuesRecursive(v, func) : func(v)])
  )

const toThreeColors = colors => mapValuesRecursive(colors, color => new Color(color))

export default toThreeColors({
  background: 'white',
  border: 'black',
  axis: {
    text: '#333',
    stroke: '#333',
  },
  series: {
    line: {
      stroke: '#1890ff',
    },
    box: {
      fill: '#777',
      medianStroke: '#111',
      minMaxStroke: '#111',
      whiskerStroke: '#222',
    },
    scatter: {
      fill: '#0050b3',
    },
  },
})
