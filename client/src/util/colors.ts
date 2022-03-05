// From https://ant.design/docs/spec/colors
import colors from '../components/charts/colors'

export default {
  blue6: '#1890ff',
  magenta6: '#eb2f96',
  gray9: '#434343',
  gray10: '#262626',
}

// From matplotlib color sequence: https://stackoverflow.com/a/42091037/780425
export const chartColors = [
  '#1f77b4',
  '#ff7f0e',
  '#2ca02c',
  '#d62728',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf',
]

export const getChartColor = seriesIndex => chartColors[seriesIndex % chartColors.length]
