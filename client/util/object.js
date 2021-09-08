// Deep-copy object
export const clone = object => JSON.parse(JSON.stringify(object))

export const deepEquals = (objectA, objectB) => JSON.stringify(objectA) === JSON.stringify(objectB)
