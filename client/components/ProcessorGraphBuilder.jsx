import React, { useRef, useState } from 'react'
import Processor from './Processor'
import { clone } from '../util/object'
import { isEventContainedInElement, isEventToLeftOfElement } from '../util/dom'

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
  // e.g. `draggingToIndices := [2,3]` means the 4th (0-indexed) parallel processor in the 3rd serial processor
  const [draggingToIndices, setDraggingToIndices] = useState(undefined)
  const processorGraphRef = useRef(undefined)

  const processors = [...selectedProcessors]
  if (draggingFrom && draggingToIndices?.length === 2) {
    const { processorDefinitionIndex, processorGraphIndices } = draggingFrom
    if (processorDefinitionIndex !== undefined) {
      const item = clone(processorDefinitions[processorDefinitionIndex])
      processors.splice(draggingToIndices[0], 0, item)
    } else if (processorGraphIndices?.length === 2) {
      const [removed] = processors.splice(processorGraphIndices[0], 1)
      processors.splice(draggingToIndices[0], 0, removed)
    }
  }

  const updateDraggingToIndices = newDraggingToIndices => {
    if (!draggingToIndices || !newDraggingToIndices) {
      setDraggingToIndices(newDraggingToIndices)
      return
    }

    const [serialIndex, parallelIndex] = newDraggingToIndices
    const [currentSerialIndex, currentParallelIndex] = draggingToIndices
    if (serialIndex !== currentSerialIndex || parallelIndex !== currentParallelIndex) {
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
        <label style={{ fontSize: '17px', fontWeight: 'bold', margin: '0px 8px' }}>Processors</label>
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
          if (!isEventContainedInElement(event, processorGraphRef.current)) {
            updateDraggingToIndices(undefined)
          }
        }}
        onDragOver={event => {
          event.preventDefault()
          const processorElements = [...processorGraphRef.current.getElementsByClassName('processor')]
          const match = processorElements
            .map((processorElement, i) => [processorElement, i])
            .find(([processorElement]) => isEventToLeftOfElement(event, processorElement))
          if (match) {
            const [processorElement, index] = match
            if (processorElement) {
              updateDraggingToIndices([index, 0])
            }
          } else {
            updateDraggingToIndices([processorElements.length, 0])
          }
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
            {processors.map((parallelProcessors, serialIndex) => {
              if (!Array.isArray(parallelProcessors)) parallelProcessors = [parallelProcessors]

              return (
                <div
                  key={`parallel${serialIndex}`}
                  style={{ display: 'flex', flexDirection: 'column', width: 'fit-content' }}
                >
                  {parallelProcessors.map((processor, parallelIndex) => {
                    const { name } = processor
                    if (serialIndex === draggingToIndices?.[0]) {
                      // Dragging placeholder definition
                      return <ProcessorDefinition key={serialIndex} name={name} />
                    }

                    return (
                      <Processor
                        key={`processor${serialIndex}`}
                        className="processor"
                        processor={processor}
                        estimatedParams={estimatedParams?.[serialIndex]}
                        onChange={(paramName, newValue) => {
                          const newSelectedProcessors = clone(selectedProcessors)
                          newSelectedProcessors[serialIndex].params[paramName] = newValue
                          onChange(newSelectedProcessors)
                        }}
                        onClose={() => {
                          const newSelectedProcessors = clone(selectedProcessors)
                          newSelectedProcessors.splice(serialIndex, 1)
                          onChange(newSelectedProcessors)
                        }}
                        onDragStart={() => {
                          setDraggingFrom({ processorGraphIndices: [serialIndex, parallelIndex] })
                        }}
                        style={{
                          margin: 5,
                          padding: 5,
                          border: '1px solid black',
                          borderRadius: 5,
                          background: 'white',
                        }}
                      />
                    )
                  })}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </>
  )
}
