import React from 'react'
import { Slider as AntSlider, InputNumber } from 'antd'

interface Props {
  name: string
  value: number
  minValue: number
  maxValue: number
  onChange?: (number) => void
  logScale?: boolean
}

export default function Slider({ name, value, minValue, maxValue, logScale = false, onChange }: Props) {
  // `position` vars correspond to slider position. (e.g. 0-1)
  // `value` vars correspond to scaled parameter values (e.g. frequency in Hz)
  const minPosition = 0.0
  const maxPosition = 1.0
  let scale
  let position
  if (logScale) {
    scale = (Math.log(maxValue) - Math.log(minValue)) / (maxPosition - minPosition)
    position = (Math.log(value) - Math.log(minValue)) / scale + minPosition
  } else {
    scale = (maxValue - minValue) / (maxPosition - minPosition)
    position = (value - minValue) / scale + minPosition
  }

  const isPreview = !onChange
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      {!isPreview && (
        <label style={{ marginRight: 4 }} htmlFor={name}>
          {name}
        </label>
      )}
      <AntSlider
        value={position}
        min={minPosition}
        max={maxPosition}
        step={(maxPosition - minPosition) / 1_000.0} // as continuous as possible
        disabled={isPreview}
        onChange={position => {
          const newValue = logScale
            ? Math.exp(Math.log(minValue) + scale * (position - minPosition))
            : minValue + scale * (position - minPosition)
          onChange(newValue)
        }}
        style={{ width: 100 }}
      />
      <InputNumber
        disabled={isPreview}
        min={minValue}
        max={maxValue}
        onChange={onChange}
        value={value.toFixed(3)}
        step={(maxValue - minValue) / 20}
        style={{ color: isPreview ? '#aaa' : '#000', marginLeft: '4px' }}
      />
    </div>
  )
}
