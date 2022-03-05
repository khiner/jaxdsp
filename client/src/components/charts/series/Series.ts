export interface Dimensions {
  x?: number
  y?: number
  width?: number
  height?: number
}

export default interface Series {
  series
  dimensions: Dimensions
  renderOrder?: number
  strokeWidth?: number
}

export interface SeriesDatum {
  x: number
  y: number
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
  numValues?: number
}

export type SeriesSummaryData = SeriesSummaryDatum[]

export type Domain = [number, number]

export interface InnerSeries {
  id: string
  label: string
  color: string
  data: SeriesData
  summaryData?: SeriesSummaryData
  xDomain?: Domain
  yDomain?: Domain
  permanent?: boolean
}

export interface Data {
  xDomain: Domain
  yDomain: Domain
  data: InnerSeries[]
}

export interface Chart {
  data
  dimensions?: Dimensions // This is actually required by the chart component, but it can be missing as a child of ChartContext.
  axes?: any[]
  xAxisHeight?: number
  yAxisHeight?: number
  yAxisWidth?: number
  fontSize?: number
  renderOrder?: number
}
