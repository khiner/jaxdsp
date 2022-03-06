import { AxisSide } from './Axis'

export interface Dimensions {
  x?: number
  y?: number
  width?: number
  height?: number
}

export interface SeriesDatum {
  x?: number
  y?: number
  x1?: number
  x2?: number
}

export type SeriesData = SeriesDatum[]

export interface SeriesSummaryDatum {
  x1: number
  x2: number
  count: number
  min: number
  p25: number
  median: number
  p75: number
  max: number
  values?: number[]
}

export type SeriesSummaryData = SeriesSummaryDatum[]

export interface Series {
  id: string
  label: string
  color: string
  data: SeriesData
  summaryData?: SeriesSummaryData
}

export type Domain = [number, number]

export interface SeriesProps {
  series: Series
  dimensions: Dimensions
  xDomain: Domain
  yDomain: Domain
  renderOrder?: number
  strokeWidth?: number
}

export interface Data {
  xDomain: Domain
  yDomain: Domain
  allSeries: Series[]
}

export default interface Chart {
  data: Data
  dimensions?: Dimensions // This is required by the chart component, but it can be missing as a child of ChartContext.
  title?: string
  axes?: AxisSide[]
  xAxisHeight?: number
  yAxisHeight?: number
  yAxisWidth?: number
  paddingTop?: number
  fontSize?: number
  renderOrder?: number
  grid?: boolean
  legend?: boolean
}
