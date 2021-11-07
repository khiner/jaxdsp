export const POSITIONS_PER_RECTANGLE = 6

// For brevity throughout,
// `ps`: positions; Float32BufferAttribute of positions
// `cs`: colors; Float32BufferAttribute of colors
// `i`: positions/colors start index

export const setPosition = (ps, cs, i, x, y, color) => {
  ps.setXYZ(i, x, y, 0)
  if (cs && color) {
    const { r, g, b } = color
    cs.setXYZ(i, r, g, b)
  }
}

export const setRectangleVertices = (ps, cs, i, x, y, width, height, color) => {
  setPosition(ps, cs, i, x, y, color)
  setPosition(ps, cs, i + 1, x + width, y, color)
  setPosition(ps, cs, i + 2, x + width, y + height, color)
  setPosition(ps, cs, i + 3, x + width, y + height, color)
  setPosition(ps, cs, i + 4, x, y + height, color)
  setPosition(ps, cs, i + 5, x, y, color)
  return i + 6
}

export const setHorizontalLineVertices = (ps, cs, startIndex, x1, x2, y, strokeWidth, strokeColor) =>
  setRectangleVertices(ps, cs, startIndex, x1, y - strokeWidth / 2, x2 - x1, strokeWidth, strokeColor)

export const setVerticalLineVertices = (ps, cs, startIndex, y1, y2, x, strokeWidth, strokeColor) =>
  setRectangleVertices(ps, cs, startIndex, x - strokeWidth / 2, y1, strokeWidth, y2 - y1, strokeColor)
