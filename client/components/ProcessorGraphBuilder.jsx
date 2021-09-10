import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { PlusOutlined } from '@ant-design/icons'
import Processor from './Processor'
import { clone } from '../util/object'

// From https://ant.design/docs/spec/colors
const colors = {
  blue6: '#1890ff',
  magenta6: '#eb2f96',
  gray9: '#434343',
  gray10: '#262626',
}

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

function ProcessorDefinition({ name, onDragStart }) {
  return (
    <div
      key={name}
      style={{
        margin: 5,
        padding: 7,
        border: `1px solid ${colors.gray9}`,
        borderRadius: 5,
        background: 'white',
        textAlign: 'middle',
      }}
      draggable={!!onDragStart}
      onDragStart={onDragStart}
    >
      {name}
    </div>
  )
}

// I miss typescript...
const ORIENTATION_HORIZONTAL = 0
const ORIENTATION_VERTICAL = 1

function ProcessorPlaceholder({ orientation }) {
  return (
    <div
      className="processor placeholder"
      style={{
        ...(orientation === ORIENTATION_HORIZONTAL
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
}

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

export default function ProcessorGraphBuilder({
  processorDefinitions,
  selectedProcessors,
  estimatedParams,
  onChange,
}) {
  const [draggingFrom, setDraggingFrom] = useState(undefined)
  // E.g. `draggingToIndices := [2,3]` means the 4th (0-indexed) parallel processor in the 3rd serial processor
  const [draggingToIndices, setDraggingToIndices] = useState(undefined)
  const [draggingToSerialIndex, draggingToParallelIndex] = draggingToIndices || [undefined, undefined]
  const [processors, setProcessors] = useState([])
  const [connections, setConnections] = useState([])
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

  useEffect(() => {
    const processors = clone(selectedProcessors.map(selectedProcessor => wrapInArray(selectedProcessor)))
    if (draggingFrom && (draggingToSerialIndex !== undefined || draggingToParallelIndex !== undefined)) {
      const { processorDefinitionIndex, processorGraphIndices } = draggingFrom
      let processor
      if (processorGraphIndices?.length === 2) {
        // Moving another processor in the graph
        const [fromSerialIndex, fromParallelIndex] = processorGraphIndices
        processor = processors[fromSerialIndex].splice(fromParallelIndex, 1)[0]
        if (processors[fromSerialIndex].length === 0) processors.splice(fromSerialIndex, 1)
      } else {
        // Creating a new processor by dragging its label
        processor = clone(processorDefinitions[processorDefinitionIndex])
      }

      if (draggingToParallelIndex === -1 || processors.length === 0) {
        processors.splice(draggingToSerialIndex, 0, wrapInArray(processor))
      } else {
        if (draggingToSerialIndex === processors.length) processors.push([])
        processors[draggingToSerialIndex].splice(draggingToParallelIndex, 0, processor)
      }
    }
    // Invariant: `processors` is an array of arrays, with no empty sub-arrays
    setProcessors(processors)
  }, [selectedProcessors, draggingFrom, draggingToIndices])

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
  }, [processors])

  console.log('render')

  return (
    <>
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
        onDragLeave={event => {
          event.preventDefault()
          if (
            !isEventContainedInElement(event, processorGraphRef.current) &&
            draggingFrom?.processorDefinitionIndex !== undefined
          ) {
            updateDraggingToIndices(undefined)
          }
        }}
        onDragOver={event => {
          event.preventDefault()
          const [currentToSerialIndex, currentToParallelIndex] = draggingToIndices || [undefined, undefined]

          const parallelProcessorElements = [
            ...processorGraphRef.current.getElementsByClassName('parallelProcessor'),
          ]
          for (let serialIndex = 0; serialIndex < parallelProcessorElements.length; serialIndex++) {
            const parallelProcessorElement = parallelProcessorElements[serialIndex]
            const { clientX } = event
            const rect = parallelProcessorElement.getBoundingClientRect()
            const { left, right, width } = rect
            if (
              clientX >= left &&
              clientX < left + width &&
              serialIndex === currentToSerialIndex &&
              currentToParallelIndex === -1
            ) {
              return
            }

            // Insert as a parallel (vertically oriented) sub-processor if
            // mouse is in the middle 1/2 of the processor's width.
            if (clientX >= left + width / 4 && clientX < right - width / 4) {
              const processorElements = [
                ...parallelProcessorElement.getElementsByClassName('processor.final'),
              ]
              const insertAboveIndex = processorElements
                .map((element, i) => [element, i])
                .find(([element]) => {
                  const { clientY } = event
                  const rect = element.getBoundingClientRect()
                  const { top, height } = rect
                  return clientY < top + height / 2
                })?.[1]
              const { processorDefinitionIndex } = draggingFrom
              const newToSerialIndex =
                serialIndex > currentToSerialIndex && processorDefinitionIndex !== undefined
                  ? serialIndex - 1
                  : serialIndex
              let newToParallelIndex =
                insertAboveIndex !== undefined ? insertAboveIndex : processorElements.length
              updateDraggingToIndices([newToSerialIndex, newToParallelIndex])
              return
            }
          }

          const insertToLeftOfIndex = parallelProcessorElements
            .map((element, i) => [element, i])
            .find(([element]) => {
              const { clientX } = event
              const rect = element.getBoundingClientRect()
              const { left, width } = rect
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
          if (processorDefinitionIndex !== undefined || processorGraphIndices) {
            onChange(processors)
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
        {processors.length === 0 && <i style={{ margin: '8px' }}>Drop processors here</i>}
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
                  const isPreview =
                    serialIndex === draggingToSerialIndex &&
                    (draggingToParallelIndex === -1 || parallelIndex === draggingToParallelIndex)
                  if (isPreview) {
                    return (
                      <ProcessorPlaceholder
                        key={key}
                        orientation={
                          draggingToParallelIndex === -1 ? ORIENTATION_VERTICAL : ORIENTATION_HORIZONTAL
                        }
                      />
                    )
                  }
                  return (
                    <Processor
                      key={key}
                      className="processor final"
                      processor={processor}
                      estimatedParams={estimatedParams?.[serialIndex]}
                      onChange={(paramName, newValue) => {
                        const newSelectedProcessors = clone(selectedProcessors)
                        newSelectedProcessors[serialIndex][parallelIndex].params[paramName] = newValue
                        onChange(newSelectedProcessors)
                      }}
                      onClose={() => {
                        const newSelectedProcessors = clone(selectedProcessors)
                        newSelectedProcessors[serialIndex].splice(parallelIndex, 1)
                        if (newSelectedProcessors[serialIndex].length === 0) {
                          newSelectedProcessors.splice(serialIndex, 1)
                        }
                        onChange(newSelectedProcessors)
                      }}
                      onDragStart={() => {
                        setDraggingFrom({ processorGraphIndices: [serialIndex, parallelIndex] })
                      }}
                      onDragEnter={() => {
                        console.log(`drag enter:${serialIndex}`)
                      }}
                      onDragExit={() => {
                        console.log(`drag exit:${serialIndex}`)
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
    </>
  )
}
