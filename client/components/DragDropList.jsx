import React, { useState } from 'react'
import { Droppable, Draggable } from 'react-beautiful-dnd'

// Static lists can be copied from but not into.
// The dragged element of a static list is still shown in its place in the list while it's dragged.
export default function DragDropList({
  droppableId = 'droppable',
  direction = 'horizontal',
  isStatic = false,
  items = [{ id: '1', content: 'item 1' }],
}) {
  const isHorizontal = direction === 'horizontal'

  const itemStyle = {
    userSelect: 'none',
    padding: 16,
    margin: isHorizontal ? '0 8px 0 0' : '0 0 8px 0',
  }

  return (
    <Droppable droppableId={droppableId} direction={direction}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          style={{
            background: !isStatic && snapshot.isDraggingOver ? 'lightblue' : 'lightgrey',
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
                <>
                  <div
                    ref={provided.innerRef}
                    {...provided.draggableProps}
                    {...provided.dragHandleProps}
                    style={{
                      ...provided.draggableProps.style, // needed on all draggable items
                      ...itemStyle,
                      background: snapshot.isDragging ? 'lightgreen' : 'grey',
                      ...(isStatic
                        ? {
                            transform: snapshot.isDragging
                              ? provided.draggableProps.style?.transform
                              : 'translate(0px, 0px)',
                          }
                        : {}),
                    }}
                  >
                    {item.content}
                  </div>
                  {isStatic && snapshot.isDragging && (
                    <div style={{ ...itemStyle, background: 'grey', transform: 'none !important' }}>
                      {item.content}
                    </div>
                  )}
                </>
              )}
            </Draggable>
          ))}
          <span style={isStatic ? { display: 'none' } : {}}>{provided.placeholder}</span>
        </div>
      )}
    </Droppable>
  )
}
