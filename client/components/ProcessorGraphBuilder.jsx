import React from 'react'
import Processor from './Processor'
import { clone } from '../util/object'

const ALL_PROCESSORS_ID = 'allProcessors'
const PROCESSOR_GRAPH_ID = 'selectedProcessors'

export default function ProcessorGraphBuilder({
  processorDefinitions,
  selectedProcessors,
  estimatedParams,
  onChange,
}) {
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
          <div
            key={name}
            style={{ margin: 5, padding: 5, border: '1px solid black', borderRadius: 5, background: 'white' }}
            draggable
            onDragStart={e => {
              const { dataTransfer } = e
              dataTransfer.setData('sourceId', ALL_PROCESSORS_ID)
              dataTransfer.setData('processorIndex', i)
              dataTransfer.setData('processorName', name)
            }}
          >
            {name}
          </div>
        ))}
      </div>
      <div
        style={{ outline: '1px dashed black' }}
        onDragEnter={() => console.log('enter')}
        onDragOver={e => {
          e.preventDefault()
        }}
        onDrop={e => {
          e.preventDefault()
          const { dataTransfer } = e
          const sourceId = dataTransfer.getData('sourceId')
          const processorIndex = dataTransfer.getData('processorIndex')
          const processorName = dataTransfer.getData('processorName')
          if (sourceId === PROCESSOR_GRAPH_ID) {
            // const sourceIndex = 1
            // const reorderedProcessors = [...selectedProcessors]
            // const [removed] = reorderedProcessors.splice(source.index, 1)
            // reorderedProcessors.splice(destination.index, 0, removed)
            // onChange(reorderedProcessors)
          } else if (sourceId === ALL_PROCESSORS_ID) {
            const destinationIndex = selectedProcessors.length // TODO
            const newSelectedProcessors = [...selectedProcessors]
            const item = clone(processorDefinitions[processorIndex])
            newSelectedProcessors.splice(destinationIndex, 0, item)
            onChange(newSelectedProcessors)
          }
        }}
      >
        {selectedProcessors?.length === 0 && <i style={{ margin: '8px' }}>Drop processors here</i>}
        {selectedProcessors?.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'row' }}>
            {selectedProcessors.map((processor, i) => (
              <Processor
                key={i}
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
                onDragStart={e => {
                  const { dataTransfer } = e
                  dataTransfer.setData('sourceId', PROCESSOR_GRAPH_ID)
                  dataTransfer.setData('processorIndex', i)
                  dataTransfer.setData('processorName', processor.name)
                }}
                style={{
                  margin: 5,
                  padding: 5,
                  border: '1px solid black',
                  borderRadius: 5,
                  background: 'white',
                }}
              />
            ))}
          </div>
        )}
      </div>
    </>
  )
}
