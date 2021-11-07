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
    this.vertexIndex = 0
  }

  setGeometryRef(geometryRef) {
    this.geometryRef = geometryRef
    if (this.geometryRef?.current) {
      this.geometryRef.current.setAttribute('position', this.positions)
      this.geometryRef.current.setAttribute('color', this.colors)
    }
  }

  start() {
    this.vertexIndex = 0
  }

  end() {
    if (this.geometryRef?.current) {
      this.geometryRef.current.setDrawRange(0, this.vertexIndex)
      this.positions.needsUpdate = true
      this.colors.needsUpdate = true
    }
  }

  addPosition(x, y, color) {
    this.positions.setXYZ(this.vertexIndex, x, y, 0)
    if (color) {
      const { r, g, b } = color
      this.colors.setXYZ(this.vertexIndex, r, g, b)
    }
    this.vertexIndex += 1
  }

  addRectangle(x, y, width, height, color) {
    this.addPosition(x, y, color)
    this.addPosition(x + width, y, color)
    this.addPosition(x + width, y + height, color)
    this.addPosition(x + width, y + height, color)
    this.addPosition(x, y + height, color)
    this.addPosition(x, y, color)
  }

  addHorizontalLine(x1, x2, y, strokeWidth, strokeColor) {
    return this.addRectangle(x1, y - strokeWidth / 2, x2 - x1, strokeWidth, strokeColor)
  }

  addVerticalLine(y1, y2, x, strokeWidth, strokeColor) {
    return this.addRectangle(x - strokeWidth / 2, y1, strokeWidth, y2 - y1, strokeColor)
  }
}
