import React, { useEffect, useRef, useState } from 'react'
import Processor from './Processor'
import { clone } from '../util/object'
import { isEventContainedInElement, isEventToLeftOfElement } from '../util/dom'

const ALL_PROCESSORS_ID = 'allProcessors'
const PROCESSOR_GRAPH_ID = 'selectedProcessors'

function ProcessorDefinition({ className, name, onDragStart }) {
  return (
    <div
      key={name}
      className={className}
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
  const [draggingToIndex, setDraggingToIndex] = useState(undefined)
  const processorGraphRef = useRef(undefined)

  const processors = [...selectedProcessors]
  if (draggingFrom && draggingToIndex !== undefined) {
    const { index: draggingFromIndex, sourceId } = draggingFrom
    if (sourceId === ALL_PROCESSORS_ID) {
      const item = clone(processorDefinitions[draggingFromIndex])
      processors.splice(draggingToIndex, 0, item)
    } else {
      // const sourceIndex = 1
      // const [removed] = newSelectedProcessorsEditing.splice(source.index, 1)
      // newSelectedProcessorsEditing.splice(destination.index, 0, removed)
    }
  }

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
            onDragStart={e => {
              setDraggingFrom({ sourceId: ALL_PROCESSORS_ID, index: i })
            }}
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
            setDraggingToIndex(undefined)
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
            if (processorElement && !processorElement.className.includes('definition')) {
              setDraggingToIndex(index)
            }
          } else {
            setDraggingToIndex(selectedProcessors.length)
          }
        }}
        onDrop={event => {
          event.preventDefault()
          if (draggingFrom === undefined || draggingToIndex === undefined) return

          const { sourceId } = draggingFrom
          if (sourceId === PROCESSOR_GRAPH_ID) {
          } else if (sourceId === ALL_PROCESSORS_ID) {
            onChange(processors)
            setDraggingToIndex(undefined)
          }
        }}
      >
        {processors.length === 0 && <i style={{ margin: '8px' }}>Drop processors here</i>}
        {processors.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
            {processors.map((processor, i) => {
              const { name } = processor
              if (i === draggingToIndex) {
                // Dragging placeholder definition
                return <ProcessorDefinition key={i} name={name} />
              }

              return (
                <Processor
                  key={i}
                  className="processor"
                  processor={processor}
                  estimatedParams={estimatedParams?.[i]}
                  onChange={(paramName, newValue) => {
                    const newSelectedProcessors = clone(selectedProcessors)
                    newSelectedProcessors[i].params[paramName] = newValue
                    onChange(newSelectedProcessors)
                  }}
                  onClose={() => {
                    const newSelectedProcessors = clone(selectedProcessors)
                    newSelectedProcessors.splice(i, 1)
                    onChange(newSelectedProcessors)
                  }}
                  onDragStart={event => {
                    setDraggingFrom({ sourceId: PROCESSOR_GRAPH_ID, index: i })
                  }}
                  onDragEnter={() => {
                    // event.preventDefault()
                    // setDraggingOverProcessor(i)
                  }}
                  onDragOver={event => {
                    event.preventDefault()
                  }}
                  onDragLeave={() => {
                    // event.preventDefault()
                    // setDraggingOverProcessor(undefined)
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
        )}
      </div>
    </>
  )
}
