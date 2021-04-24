import React, { useState } from 'react'
import { Droppable, Draggable } from 'react-beautiful-dnd'

// Static lists can be copied from but not into.
// The dragged element of a static list is still shown in its place in the list while it's dragged.
export default function DragDropList({
  droppableId = 'droppable',
  direction = 'horizontal',
  isStatic = false,
  items = [{ id: '1', content: 'item 1' }],
  style = {},
  draggingStyle = {},
  itemStyle = {},
  itemDraggingStyle = {},
  emptyContent = <div>Drop items here</div>,
}) {
  const isHorizontal = direction === 'horizontal'

  style = {
    ...{
      display: 'flex',
      flexDirection: isHorizontal ? 'row' : 'column',
      padding: 8,
      overflow: 'auto',
    },
    ...style,
  }
  draggingStyle = {
    ...{
      background: 'lightblue',
    },
    ...draggingStyle,
  }

  itemStyle = {
    ...{
      background: 'white',
      border: '1px solid black',
      borderRadius: '4px',
      userSelect: 'none',
      padding: 8,
      margin: isHorizontal ? '0 8px 0 0' : '0 0 8px 0',
    },
    ...itemStyle,
  }
  itemDraggingStyle = {
    border: '1px dashed black',
    background: 'lightgreen',
    ...itemDraggingStyle,
  }

  return (
    <Droppable droppableId={droppableId} direction={direction}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          style={{ ...style, ...(!isStatic && snapshot.isDraggingOver ? draggingStyle : {}) }}
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
                      ...(isStatic
                        ? {
                            transform: snapshot.isDragging
                              ? provided.draggableProps.style?.transform
                              : 'translate(0px, 0px)',
                          }
                        : {}),
                      ...(snapshot.isDragging ? itemDraggingStyle : {}),
                    }}
                  >
                    {item.content}
                  </div>
                  {isStatic && snapshot.isDragging && (
                    <div style={{ ...itemStyle, transform: 'none !important' }}>{item.content}</div>
                  )}
                </>
              )}
            </Draggable>
          ))}
          <span style={isStatic ? { display: 'none' } : {}}>{provided.placeholder}</span>
          {items?.length === 0 && !snapshot.isDraggingOver && emptyContent}
        </div>
      )}
    </Droppable>
  )
}
