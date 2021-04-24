import React, { useState } from 'react'
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd'

export default function DraggableList({ direction = 'horizontal', numItems = 6 }) {
  const [items, setItems] = useState(
    [...Array(numItems).keys()].map(k => ({
      id: `item-${k}`,
      content: `item ${k}`,
    }))
  )

  const isHorizontal = direction === 'horizontal'

  return (
    <DragDropContext
      onDragEnd={result => {
        if (result.destination) {
          const reorderedItems = [...items]
          const [removed] = reorderedItems.splice(result.source.index, 1)
          reorderedItems.splice(result.destination.index, 0, removed)
          setItems(reorderedItems)
        }
      }}
    >
      <Droppable droppableId="droppable">
        {(provided, snapshot) => (
          <div
            ref={provided.innerRef}
            style={{
              background: snapshot.isDraggingOver ? 'lightblue' : 'lightgrey',
              display: 'flex',
              flexDirection: isHorizontal ? 'row' : 'column',
              padding: 8,
              overflow: 'auto',
              ...(isHorizontal ? {} : { width: 250 }),
            }}
            {...provided.droppableProps}
          >
            {items.map((item, index) => (
              <Draggable key={item.id} draggableId={item.id} index={index}>
                {(provided, snapshot) => (
                  <div
                    ref={provided.innerRef}
                    {...provided.draggableProps}
                    {...provided.dragHandleProps}
                    style={{
                      userSelect: 'none',
                      padding: 16,
                      margin: isHorizontal ? '0 8px 0 0' : '0 0 8px 0',
                      background: snapshot.isDragging ? 'lightgreen' : 'grey',
                      ...provided.draggableProps.style, // needed on all draggable items
                    }}
                  >
                    {item.content}
                  </div>
                )}
              </Draggable>
            ))}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </DragDropContext>
  )
}
