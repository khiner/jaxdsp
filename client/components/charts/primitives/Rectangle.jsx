export const VERTICES_PER_POSITION = 3
export const POSITIONS_PER_RECTANGLE = 6

// For brevity throughout,
// `ps`: positions; array of position floats
// `cs`: colors; array of color floats.
// `i`: positions/colors start index

export const setPosition = (ps, cs, i, x, y, color) => {
  ps[i] = x
  ps[i + 1] = y
  ps[i + 2] = 0

  if (cs && color) {
    const { r, g, b } = color
    cs[i] = r
    cs[i + 1] = g
    cs[i + 2] = b
  }

  return i + VERTICES_PER_POSITION
}

export const addRectangleVertices = (ps, cs, i, x, y, width, height, color) => {
  i = setPosition(ps, cs, i, x, y, color)
  i = setPosition(ps, cs, i, x + width, y, color)
  i = setPosition(ps, cs, i, x + width, y + height, color)
  i = setPosition(ps, cs, i, x + width, y + height, color)
  i = setPosition(ps, cs, i, x, y + height, color)
  i = setPosition(ps, cs, i, x, y, color)
  return i
}

export const addHorizontalLineVertices = (ps, cs, startIndex, x1, x2, y, strokeWidth, strokeColor) =>
  addRectangleVertices(ps, cs, startIndex, x1, y - strokeWidth / 2, x2 - x1, strokeWidth, strokeColor)

export const addVerticalLineVertices = (ps, cs, startIndex, y1, y2, x, strokeWidth, strokeColor) =>
  addRectangleVertices(ps, cs, startIndex, x - strokeWidth / 2, y1, strokeWidth, y2 - y1, strokeColor)
