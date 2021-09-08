import React, { useRef, useState } from 'react'
import Processor from './Processor'
import { clone } from '../util/object'

const isEventContainedInElement = (event, element) => {
  if (!event || !element) return false

  const { clientX, clientY } = event
  const rect = element.getBoundingClientRect()
  return clientY >= rect.top && clientY < rect.bottom && clientX >= rect.left && clientX < rect.right
}

const wrapInArray = itemOrArray => (Array.isArray(itemOrArray) ? itemOrArray : [itemOrArray])

function ProcessorDefinition({ name, onDragStart }) {
  return (
    <div
      key={name}
      style={{
        margin: 5,
        padding: 5,
        border: '1px solid black',
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

  const processorGraphRef = useRef(undefined)

  const processors = clone(selectedProcessors.map(selectedProcessor => wrapInArray(selectedProcessor)))
  if (draggingFrom && (draggingToSerialIndex !== undefined || draggingToParallelIndex !== undefined)) {
    const { processorDefinitionIndex, processorGraphIndices } = draggingFrom
    let processor
    if (processorGraphIndices?.length === 2) {
      // Moving another processor in the graph
      const [fromSerialIndex, fromParallelIndex] = processorGraphIndices
      processor = processors[fromSerialIndex].splice(fromParallelIndex, 1)[0]
      if (processors[fromSerialIndex].length === 0) processors.splice(fromSerialIndex, 1)
      if (draggingToSerialIndex === processors.length) processors.push([])
    } else {
      // Creating a new processor by dragging its label
      processor = clone(processorDefinitions[processorDefinitionIndex])
    }

    if (draggingToParallelIndex === -1 || processors.length === 0) {
      processors.splice(draggingToSerialIndex, 0, wrapInArray(processor))
    } else {
      processors[draggingToSerialIndex].splice(draggingToParallelIndex, 0, processor)
    }
  }
  // Invariant: `processors` is an array of arrays, with no empty subarrays

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
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: 'fit-content',
          minWidth: 200,
          outline: '1px dashed black',
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
            const { left, right } = rect
            const width = right - left
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
              const processorElements = [...parallelProcessorElement.getElementsByClassName('processor')]
              const insertAboveIndex = processorElements
                .map((element, i) => [element, i])
                .find(([element]) => {
                  const { clientY } = event
                  const rect = element.getBoundingClientRect()
                  const { top, bottom } = rect
                  const height = bottom - top
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
              const { left, right } = rect
              const width = right - left
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
        {processors.length === 0 && <i style={{ margin: '8px' }}>Drop processors here</i>}
        {processors.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
            {processors.map((parallelProcessors, serialIndex) => (
              <div
                key={`parallel${serialIndex}`}
                className="parallelProcessor"
                style={{ display: 'flex', flexDirection: 'column', width: 'fit-content' }}
              >
                {parallelProcessors.map((processor, parallelIndex) => {
                  const key = `processor-${serialIndex}-${parallelIndex}`
                  const isPreview =
                    serialIndex === draggingToSerialIndex &&
                    (draggingToParallelIndex === -1 || parallelIndex === draggingToParallelIndex)
                  return (
                    <Processor
                      key={key}
                      className="processor"
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
                        background: isPreview ? 'grey' : 'white',
                        margin: 5,
                        padding: 7,
                        border: '1px solid black',
                        borderRadius: 5,
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
