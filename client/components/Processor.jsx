import React from 'react'

import Slider from './Slider'
import { snakeCaseToSentence } from '../util/string'

export default function Processor({ processor, isEstimatingParams, trainState, mouseX, onChange }) {
  return (
    <>
      <label>{processor.name}</label>
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
              mouseX={mouseX}
            />
          ))}
        </div>
        {isEstimatingParams && trainState?.params && (
          <div>
            {processor.param_definitions.map(
              ({ name, min_value, max_value, log_scale }) =>
                !isNaN(trainState.params[name]) && (
                  <Slider
                    key={name}
                    name={snakeCaseToSentence(name)}
                    value={trainState.params[name]}
                    minValue={min_value}
                    maxValue={max_value}
                    logScale={log_scale}
                    onChange={null}
                    mouseX={mouseX}
                  />
                )
            )}
          </div>
        )}
      </div>
    </>
  )
}
