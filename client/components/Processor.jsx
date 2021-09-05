import React from 'react'
import { Button } from 'antd'
import { CloseOutlined } from '@ant-design/icons'

import Slider from './Slider'
import { snakeCaseToSentence } from '../util/string'

export default function Processor({
  processor,
  estimatedParams,
  onChange,
  onClose,
  onDragStart,
  className,
  style,
}) {
  return (
    <div draggable={!!onDragStart} onDragStart={onDragStart} className={className} style={style}>
      <div>
        <label style={{ fontSize: 16, fontWeight: 'bold', marginRight: 4 }}>{processor.name}</label>
        <Button
          style={{ float: 'right', marginTop: -5 }}
          type="link"
          icon={<CloseOutlined />}
          onClick={onClose}
        />
      </div>
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        <div>
          {processor.param_definitions.map(({ name, default_value, min_value, max_value, log_scale }) => (
            <Slider
              key={name}
              name={snakeCaseToSentence(name)}
              value={processor.params[name] || default_value || 0.0}
              minValue={min_value}
              maxValue={max_value}
              logScale={log_scale}
              onChange={newValue => onChange(name, newValue)}
            />
          ))}
        </div>
        {estimatedParams && (
          <div>
            {processor.param_definitions.map(
              ({ name, min_value, max_value, log_scale }) =>
                !isNaN(estimatedParams[name]) && (
                  <Slider
                    key={name}
                    name={snakeCaseToSentence(name)}
                    value={estimatedParams[name]}
                    minValue={min_value}
                    maxValue={max_value}
                    logScale={log_scale}
                    onChange={null}
                  />
                )
            )}
          </div>
        )}
      </div>
    </div>
  )
}
