import React, { useLayoutEffect, useRef, useState } from 'react'
import { PlusOutlined } from '@ant-design/icons'
import { deepCopy } from '../util/object'
import Processor from './Processor'
import colors from '../util/colors'

const wrapInArray = itemOrArray => (Array.isArray(itemOrArray) ? itemOrArray : [itemOrArray])

const isEventContainedInElement = (event, element) => {
  if (!event || !element) return false

  const { clientX, clientY } = event
  const rect = element.getBoundingClientRect()
  return clientY >= rect.top && clientY < rect.bottom && clientX >= rect.left && clientX < rect.right
}

function Connection({ beginX, beginY, endX, endY }) {
  const midX = beginX + (endX - beginX) / 2
  return <path d={`M ${beginX} ${beginY} C ${midX} ${beginY} ${midX} ${endY} ${endX} ${endY}`} />
}

interface ProcessorDefinitionProps {
  name: string
  onDragStart: () => void
}

const ProcessorDefinition = ({ name, onDragStart }: ProcessorDefinitionProps) => (
  <div
    key={name}
    style={{
      margin: 5,
      padding: 7,
      border: `1px solid ${colors.gray9}`,
      borderRadius: 5,
      background: 'white',
      textAlign: 'center', // TODO not needed?
    }}
    draggable={!!onDragStart}
    onDragStart={onDragStart}
  >
    {name}
  </div>
)

const enum ORIENTATION {
  horizontal,
  vertical,
}

interface ProcessorPlaceholderProps {
  orientation: ORIENTATION
}

const ProcessorPlaceholder = ({ orientation }: ProcessorPlaceholderProps) => (
  <div
    className="processor placeholder"
    style={{
      ...(orientation === ORIENTATION.horizontal
        ? { minWidth: 80, height: '3em' }
        : {
            minHeight: 80,
            width: '3em',
            height: '100%',
          }),
      alignSelf: 'stretch',
      padding: 5,
      background: 'white',
      border: `2px solid ${colors.magenta6}`,
      borderRadius: 5,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      margin: '0 10px',
    }}
  >
    <PlusOutlined style={{ color: colors.blue6, fontSize: 20 }} />
  </div>
)

const getRelativeRect = (rect, relativeToRect) => {
  if (!rect || !relativeToRect) return undefined

  const { top, bottom, left, right, width, height } = rect
  return {
    top: top - relativeToRect.top,
    bottom: bottom - relativeToRect.top,
    left: left - relativeToRect.left,
    right: right - relativeToRect.left,
    width,
    height,
  }
}

interface Props {
  processorDefinitions: any[]
  selectedProcessors: any[]
  estimatedParams: any[]
  onChange: (any) => void
}

function ProcessorGraphBuilder({
  processorDefinitions,
  selectedProcessors,
  estimatedParams,
  onChange,
}: Props) {
  const [draggingFrom, setDraggingFrom] = useState(undefined)
  // E.g. `draggingToIndices := [2,3]` means the 4th (0-indexed) parallel processor in the 3rd serial processor
  const [draggingToIndices, setDraggingToIndices] = useState(undefined)
  const [draggingToSerialIndex, draggingToParallelIndex] = draggingToIndices || [undefined, undefined]
  const [connections, setConnections] = useState([])
  const parentRef = useRef(undefined)
  const processorGraphRef = useRef(undefined)

  const updateDraggingToIndices = newDraggingToIndices => {
    if (!draggingToIndices || !newDraggingToIndices) {
      setDraggingToIndices(newDraggingToIndices)
      return
    }

    const [serialIndex, parallelIndex] = newDraggingToIndices
    if (serialIndex !== draggingToSerialIndex || parallelIndex !== draggingToParallelIndex) {
      setDraggingToIndices([serialIndex, parallelIndex])
    }
  }

  const stopDragging = () => {
    setDraggingFrom(undefined)
    updateDraggingToIndices(undefined)
  }

  const processors = deepCopy(selectedProcessors.map(selectedProcessor => wrapInArray(selectedProcessor)))
  if (draggingFrom && (draggingToSerialIndex !== undefined || draggingToParallelIndex !== undefined)) {
    const { processorDefinitionIndex, processorGraphIndices } = draggingFrom
    let processorPreview
    if (processorGraphIndices?.length === 2) {
      // Moving another processor in the graph
      const [fromSerialIndex, fromParallelIndex] = processorGraphIndices
      processorPreview = processors[fromSerialIndex].splice(fromParallelIndex, 1)[0]
      if (processors[fromSerialIndex].length === 0) processors.splice(fromSerialIndex, 1)
    } else {
      // Creating a new processor by dragging its label
      processorPreview = deepCopy(processorDefinitions[processorDefinitionIndex])
    }

    processorPreview.isPreview = true
    if (draggingToParallelIndex === -1 || processors.length === 0) {
      processors.splice(draggingToSerialIndex, 0, wrapInArray(processorPreview))
    } else {
      if (draggingToSerialIndex === processors.length) processors.push([])
      processors[draggingToSerialIndex].splice(draggingToParallelIndex, 0, processorPreview)
    }
  }
  // Invariant: `processors` is an array of arrays, with no empty sub-arrays

  useLayoutEffect(() => {
    const processorGraphRect = processorGraphRef.current.getBoundingClientRect()
    const parallelProcessorElements = [
      ...processorGraphRef.current.getElementsByClassName('parallelProcessor'),
    ]

    setConnections(
      parallelProcessorElements.flatMap((parallelProcessor, parallelIndex) =>
        [...parallelProcessor.getElementsByClassName('processor')].flatMap(serialProcessor => {
          const rect = getRelativeRect(serialProcessor.getBoundingClientRect(), processorGraphRect)
          const parentRect = getRelativeRect(
            serialProcessor.parentElement.getBoundingClientRect(),
            processorGraphRect
          )
          const innerConnections = []
          if (parallelIndex !== 0) {
            innerConnections.push({
              beginX: parentRect.left - 1,
              beginY: parentRect.top + parentRect.height / 2,
              endX: rect.left,
              endY: rect.top + rect.height / 2,
            })
          }
          if (parallelIndex !== parallelProcessorElements.length - 1) {
            innerConnections.push({
              beginX: rect.right - 1,
              beginY: rect.top + rect.height / 2,
              endX: parentRect.right,
              endY: parentRect.top + parentRect.height / 2,
            })
          }
          return innerConnections
        })
      )
    )
  }, [JSON.stringify(processors)])

  return (
    <div
      ref={parentRef}
      onDragLeave={event => {
        event.preventDefault()
        if (!isEventContainedInElement(event, parentRef.current) && draggingFrom?.processorGraphIndices) {
          setDraggingFrom(undefined)
        }
      }}
      onDragOver={event => {
        event.preventDefault()
        if (!draggingFrom || !isEventContainedInElement(event, processorGraphRef.current)) return
        if (processors?.length === 1 && processors[0]?.length === 1 && draggingFrom.processorGraphIndices) {
          updateDraggingToIndices(undefined)
          return
        }

        const [currentToSerialIndex, currentToParallelIndex] = draggingToIndices || [undefined, undefined]
        const parallelProcessorElements = [
          ...processorGraphRef.current.getElementsByClassName('parallelProcessor'),
        ]

        for (let serialIndex = 0; serialIndex < parallelProcessorElements.length; serialIndex++) {
          const parallelProcessorElement = parallelProcessorElements[serialIndex]
          const processorElements = [...parallelProcessorElement.getElementsByClassName('final')]
          const { clientX } = event
          const rect = parallelProcessorElement.getBoundingClientRect()
          const { left, right, width } = rect
          const { processorDefinitionIndex } = draggingFrom

          // Insert as a parallel (vertically oriented) sub-processor if
          // mouse is in the middle 1/2 of the processor's width.
          if (
            !(currentToSerialIndex === serialIndex && processors[currentToSerialIndex].length === 1) &&
            clientX >= left + width / 4 &&
            clientX < right - width / 4
          ) {
            const insertAboveIndex = processorElements
              .map((element, i) => [element, i])
              .find(([element]) => {
                const { clientY } = event
                const { top, height } = element.getBoundingClientRect()
                return clientY < top + height / 2
              })?.[1]

            const newToSerialIndex =
              serialIndex > currentToSerialIndex && processorDefinitionIndex !== undefined
                ? serialIndex - 1
                : serialIndex
            const newToParallelIndex =
              insertAboveIndex !== undefined ? insertAboveIndex : processorElements.length
            updateDraggingToIndices([newToSerialIndex, newToParallelIndex])
            return
          }
        }

        const insertToLeftOfIndex = parallelProcessorElements
          .map((element, i) => [element, i])
          .find(([element]) => {
            const { clientX } = event
            const { left, width } = element.getBoundingClientRect()
            return clientX < left + width / 2
          })?.[1]
        let newToSerialIndex =
          insertToLeftOfIndex !== undefined ? insertToLeftOfIndex : parallelProcessorElements.length
        if (newToSerialIndex > currentToSerialIndex && currentToParallelIndex === -1) {
          newToSerialIndex -= 1
        }
        updateDraggingToIndices([newToSerialIndex, -1])
      }}
      onDrop={event => {
        event.preventDefault()
        if (!draggingFrom || !draggingToIndices) return

        const { processorDefinitionIndex, processorGraphIndices } = draggingFrom
        stopDragging()
        if (processorDefinitionIndex !== undefined || processorGraphIndices) {
          processors.forEach(parallelProcessors =>
            parallelProcessors.forEach(processor => delete processor.isPreview)
          )
          onChange(processors)
        }
      }}
    >
      <div
        style={{
          width: 'fit-content',
          display: 'flex',
          flexDirection: 'row',
          margin: '4px',
          alignItems: 'center',
        }}
      >
        <label style={{ fontSize: '17px', fontWeight: 'bold', marginRight: '0.5em' }}>Processors:</label>
        {processorDefinitions.map(({ name }, i) => (
          <ProcessorDefinition
            key={i}
            name={name}
            onDragStart={() => setDraggingFrom({ processorDefinitionIndex: i })}
          />
        ))}
      </div>
      <div
        ref={processorGraphRef}
        style={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: 'fit-content',
          minWidth: 200,
          border: `1px dashed ${colors.gray9}`,
          borderRadius: 5,
          padding: 5,
        }}
        onDragOver={event => {
          event.preventDefault()
        }}
        onDragLeave={event => {
          event.preventDefault()
          if (!isEventContainedInElement(event, processorGraphRef.current)) {
            updateDraggingToIndices(undefined)
          }
        }}
      >
        {connections.length > 0 && (
          <svg
            preserveAspectRatio="none"
            width="100%"
            height="100%"
            style={{
              position: 'absolute',
              stroke: 'black',
              strokeWidth: '3px',
              zIndex: -1,
            }}
          >
            {connections.map((connection, i) => (
              <Connection key={`c-${i}`} {...connection} />
            ))}
          </svg>
        )}
        {processors.length === 0 && <i style={{ margin: 8 }}>Drop processors here</i>}
        {processors.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'stretch' }}>
            {processors.map((parallelProcessors, serialIndex) => (
              <div
                key={`parallel${serialIndex}`}
                className="parallelProcessor"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  width: 'fit-content',
                  gap: 10,
                }}
              >
                {parallelProcessors.map((processor, parallelIndex) => {
                  const key = `processor-${serialIndex}-${parallelIndex}`
                  if (processor.isPreview) {
                    return (
                      <ProcessorPlaceholder
                        key={key}
                        orientation={
                          draggingToParallelIndex === -1 ? ORIENTATION.vertical : ORIENTATION.horizontal
                        }
                      />
                    )
                  }
                  return (
                    <Processor
                      key={key}
                      className="processor final"
                      processor={processor}
                      estimatedParams={estimatedParams?.[serialIndex]?.[parallelIndex]}
                      onChange={(paramName, newValue) => {
                        const newSelectedProcessors = deepCopy(selectedProcessors)
                        newSelectedProcessors[serialIndex][parallelIndex].params[paramName] = newValue
                        onChange(newSelectedProcessors)
                      }}
                      onClose={() => {
                        const newSelectedProcessors = deepCopy(selectedProcessors)
                        newSelectedProcessors[serialIndex].splice(parallelIndex, 1)
                        if (newSelectedProcessors[serialIndex].length === 0) {
                          newSelectedProcessors.splice(serialIndex, 1)
                        }
                        onChange(newSelectedProcessors)
                      }}
                      onDragStart={() => {
                        setDraggingFrom({
                          processorGraphIndices: [
                            serialIndex,
                            parallelProcessors.length === 1 ? -1 : parallelIndex,
                          ],
                        })
                      }}
                      style={{
                        background: 'white',
                        padding: 7,
                        border: `1px solid ${colors.gray9}`,
                        borderRadius: 5,
                        margin: '0 10px',
                      }}
                    />
                  )
                })}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default React.memo(ProcessorGraphBuilder)
