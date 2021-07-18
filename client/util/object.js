// Deep-copy object
export function clone(object) {
  return JSON.parse(JSON.stringify(object))
}
