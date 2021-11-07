import { BufferAttribute } from 'three'

export const POSITIONS_PER_RECTANGLE = 6

// For brevity throughout,
// `ps`: positions; Float32BufferAttribute of positions
// `cs`: colors; Float32BufferAttribute of colors
// `i`: positions/colors start index

export default class Vertices {
  constructor(maxLength) {
    this.positions = new BufferAttribute(new Float32Array(3 * maxLength), 3)
    this.colors = new BufferAttribute(new Float32Array(3 * maxLength), 3)
  }

  setGeometryRef(geometryRef) {
    this.geometryRef = geometryRef
    if (this.geometryRef?.current) {
      this.geometryRef.current.setAttribute('position', this.positions)
      this.geometryRef.current.setAttribute('color', this.colors)
    }
  }

  setDrawLength(drawLength) {
    if (this.geometryRef?.current) {
      this.geometryRef.current.setDrawRange(0, drawLength)
      this.positions.needsUpdate = true
      this.colors.needsUpdate = true
    }
  }

  setPosition(i, x, y, color) {
    this.positions.setXYZ(i, x, y, 0)
    if (color) {
      const { r, g, b } = color
      this.colors.setXYZ(i, r, g, b)
    }
  }

  setRectangle(i, x, y, width, height, color) {
    this.setPosition(i, x, y, color)
    this.setPosition(i + 1, x + width, y, color)
    this.setPosition(i + 2, x + width, y + height, color)
    this.setPosition(i + 3, x + width, y + height, color)
    this.setPosition(i + 4, x, y + height, color)
    this.setPosition(i + 5, x, y, color)
    return i + 6
  }

  setHorizontalLine(startIndex, x1, x2, y, strokeWidth, strokeColor) {
    return this.setRectangle(startIndex, x1, y - strokeWidth / 2, x2 - x1, strokeWidth, strokeColor)
  }

  setVerticalLine(startIndex, y1, y2, x, strokeWidth, strokeColor) {
    return this.setRectangle(startIndex, x - strokeWidth / 2, y1, strokeWidth, y2 - y1, strokeColor)
  }
}
