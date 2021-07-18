import React, { useEffect, useRef, useState } from 'react'

export default function Slider({ name, value, minValue, maxValue, logScale, onChange, mouseX }) {
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

  const [isMouseDown, setIsMouseDown] = useState(false)
  const sliderRef = useRef()

  useEffect(() => {
    if (!isMouseDown || !onChange || !sliderRef.current || !mouseX) return

    const sliderRect = sliderRef.current.getBoundingClientRect()
    const sliderX = mouseX - sliderRect.left
    const position = Math.max(0.0, Math.min(sliderX / sliderRect.width, 1.0))
    const newValue = logScale
      ? Math.exp(Math.log(minValue) + scale * (position - minPosition))
      : minValue + scale * (position - minPosition)
    return onChange(newValue)
  }, [mouseX, isMouseDown, sliderRef])

  const isPreview = !onChange
  return (
    <div style={{ display: 'flex', alignItems: 'center', margin: '5px' }}>
      {!isPreview && <label htmlFor={name}>{name}</label>}
      <input
        type="range"
        name={name}
        value={position}
        min={minPosition}
        max={maxPosition}
        step={(maxPosition - minPosition) / 10_000.0} // as continuous as possible
        disabled={isPreview}
        ref={sliderRef}
        onChange={event => {}}
        onMouseDown={() => {
          setIsMouseDown(true)
        }}
        onMouseUp={() => setIsMouseDown(false)}
      />
      <span style={{ color: isPreview ? '#aaa' : '#000', marginLeft: '4px' }}>{value.toFixed(3)}</span>
    </div>
  )
}
