export const isEventContainedInElement = (event, element) => {
  if (!event || !element) return false

  const { clientX, clientY } = event
  const rect = element.getBoundingClientRect()
  return clientY >= rect.top && clientY < rect.bottom && clientX >= rect.left && clientX < rect.right
}

export const isEventToLeftOfElement = (event, element) => {
  if (!event || !element) return false

  const { clientX } = event
  const rect = element.getBoundingClientRect()
  return clientX < rect.left + (rect.right - rect.left) / 2
}
