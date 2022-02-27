export default interface Props {
  data
  dimensions?: any // This is actually required by the chart component, but it can be missing as a child of ChartContext.
  axes?: any[]
  xAxisHeight?: number
  yAxisHeight?: number
  yAxisWidth?: number
  fontSize?: number
  renderOrder?: number
}
