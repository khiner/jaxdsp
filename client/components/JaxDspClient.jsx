import React, { useEffect, useRef, useState } from 'react'
import adapter from 'webrtc-adapter' // eslint-disable-line no-unused-vars

import DraggableList from './DraggableList'

import { negotiatePeerConnection } from '../helpers/WebRtcHelper'

let peerConnection = null
let dataChannel = null

const AUDIO_INPUT_SOURCES = {
  microphone: {
    label: 'Microphone',
  },
  testSample: {
    label: 'Test sample',
  },
}

const NO_PROCESSOR_LABEL = 'None'

// E.g. long_parameter_name => Long Parameter Name
function snakeCaseToSentence(name) {
  return name
    ?.split('_')
    .join(' ')
    .replace(/^(.)/, firstLetter => firstLetter.toUpperCase())
}

function Slider({ name, value, minValue, maxValue, logScale, onChange }) {
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
    <div key={name} style={{ display: 'flex', alignItems: 'center', margin: '5px' }}>
      {!isPreview && <label htmlFor={name}>{name}</label>}
      <input
        type="range"
        name={name}
        value={position}
        min={minPosition}
        max={maxPosition}
        step={(maxPosition - minPosition) / 100_000.0} // as continuous as possible
        onChange={event => {
          if (!onChange) return

          const position = +event.target.value
          const newValue = logScale
            ? Math.exp(Math.log(minValue) + scale * (position - minPosition))
            : minValue + scale * (position - minPosition)
          return onChange(newValue)
        }}
        disabled={isPreview}
      />
      <span style={{ color: isPreview ? '#aaa' : '#000', marginLeft: '4px' }}>{value.toFixed(3)}</span>
    </div>
  )
}

export default function JaxDspClient({ testSample }) {
  const [audioInputSourceLabel, setAudioInputSourceLabel] = useState(AUDIO_INPUT_SOURCES.testSample.label)
  const [isStreamingAudio, setIsStreamingAudio] = useState(false)
  const [isEstimatingParams, setIsEstimatingParams] = useState(false)
  const [processors, setProcessors] = useState(null)
  const [optimizers, setOptimizers] = useState(null)
  const [processor, setProcessor] = useState(null)
  const [lossOptions, setLossOptions] = useState(null)
  const [optimizer, setOptimizer] = useState(null)
  const [trainState, setTrainState] = useState({})
  const [audioStreamErrorMessage, setAudioStreamErrorMessage] = useState(null)
  const [clientUid, setClientUid] = useState(null)

  const audioRef = useRef(null)

  const sendProcessor = () => dataChannel?.send(JSON.stringify({ processor }))
  const sendOptimizer = () => dataChannel?.send(JSON.stringify({ optimizer }))
  const sendLossOptions = () => dataChannel?.send(JSON.stringify({ loss_options: lossOptions }))

  const onAudioStreamError = (displayMessage, error) => {
    setIsStreamingAudio(false)
    setAudioStreamErrorMessage(displayMessage)
    const errorMessage = error ? `${displayMessage}: ${error}` : displayMessage
    console.error(errorMessage)
  }

  useEffect(() => {
    sendProcessor()
  }, [processor])

  useEffect(() => {
    if (clientUid === null) return

    // TODO wss? (SSL)
    const ws = new WebSocket('ws://127.0.0.1:8765/')
    ws.onopen = () => {
      ws.send(JSON.stringify({ client_uid: clientUid }))
    }
    ws.onmessage = event => {
      const message = JSON.parse(event.data)
      const { train_state: trainState } = message
      if (trainState) setTrainState(trainState)
    }
    ws.onclose = event => {
      const { wasClean, code } = event
      if (!wasClean) {
        setAudioStreamErrorMessage(`WebSocket unexpectedly closed with code ${code}`)
      }
    }
    ws.onerror = () => {
      setAudioStreamErrorMessage('WebSocket connection error')
    }
    return () => {
      ws.close()
    }
  }, [clientUid])

  useEffect(() => {
    const openPeerConnection = () => {
      if (peerConnection) return

      peerConnection = new RTCPeerConnection()
      peerConnection.onconnectionstatechange = function () {
        switch (peerConnection.connectionState) {
          case 'disconnected':
          case 'failed':
            onAudioStreamError('Stream has terminated unexpectedly', null)
            break
          case 'connected': // Fully connected
          case 'closed': // Expected close
          default:
            break
        }
      }
      peerConnection.addEventListener('track', event => (audioRef.current.srcObject = event.streams[0]))
      dataChannel = peerConnection.createDataChannel('jaxdsp-client', { ordered: true })
      dataChannel.onopen = () => dataChannel.send('get_state')
      dataChannel.onmessage = event => {
        const message = JSON.parse(event.data)
        const { processor_definitions, processor, optimizer_definitions, optimizer, loss_options } = message

        if (processor_definitions) setProcessors(processor_definitions)
        if (processor) setProcessor(processor)
        if (optimizer_definitions) setOptimizers(optimizer_definitions)
        if (optimizer) setOptimizer(optimizer)
        if (lossOptions) setLossOptions(loss_options)
      }
    }

    const closePeerConnection = () => {
      if (!peerConnection) return

      peerConnection.getSenders().forEach(sender => sender?.track?.stop())
      dataChannel?.close()
      dataChannel = null
      peerConnection.getTransceivers()?.forEach(transceiver => transceiver.stop())
      peerConnection.close()
      peerConnection = null
      setProcessor(null)
      setProcessors(null)
      setOptimizer(null)
      setOptimizers(null)
      setIsEstimatingParams(false)
      setTrainState({})
      setClientUid(null)
    }

    const addOrReplaceTrack = async track => {
      const audioSender = peerConnection.getSenders().find(s => s.track?.kind === 'audio')
      if (audioSender) {
        audioSender.replaceTrack(track)
      } else {
        peerConnection.addTrack(track)
        try {
          const clientUid = await negotiatePeerConnection(peerConnection)
          setClientUid(clientUid)
        } catch (error) {
          onAudioStreamError('Failed to negotiate RTC peer connection', error)
        }
      }
    }

    const startStreamingMicrophone = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false },
          video: false,
        })
        addOrReplaceTrack(stream.getTracks()[0])
      } catch (error) {
        onAudioStreamError('Could not acquire media', error)
      }
    }

    const startStreamingTestSample = () => {
      const sampleAudio = new Audio(testSample)
      const audioContext = new AudioContext()
      const sampleSource = audioContext.createMediaElementSource(sampleAudio)
      const sampleDestination = audioContext.createMediaStreamDestination()
      sampleSource.connect(sampleDestination)
      addOrReplaceTrack(sampleDestination.stream.getAudioTracks()[0])

      sampleAudio.loop = true
      sampleAudio.currentTime = 0
      sampleAudio.play()
    }

    if (isStreamingAudio) {
      setAudioStreamErrorMessage(null)
      openPeerConnection()
      if (audioInputSourceLabel === AUDIO_INPUT_SOURCES.microphone.label) startStreamingMicrophone()
      else if (audioInputSourceLabel === AUDIO_INPUT_SOURCES.testSample.label) startStreamingTestSample()
    } else if (!isStreamingAudio) {
      closePeerConnection()
    }

    return () => {
      closePeerConnection()
    }
  }, [isStreamingAudio, audioInputSourceLabel])

  const startEstimatingParams = () => {
    setIsEstimatingParams(true)
    if (dataChannel) dataChannel.send('start_estimating_params')
  }

  const stopEstimatingParams = () => {
    setIsEstimatingParams(false)
    if (dataChannel) dataChannel.send('stop_estimating_params')
  }

  return (
    <div>
      <div>
        <span>Audio input source:</span>{' '}
        <select
          value={audioInputSourceLabel}
          onChange={event => setAudioInputSourceLabel(event.target.value)}
        >
          {Object.values(AUDIO_INPUT_SOURCES).map(({ label }) => (
            <option key={label} value={label}>
              {label}
            </option>
          ))}
        </select>
      </div>
      {isStreamingAudio && (
        <>
          {processors && (
            <div>
              <select
                value={processor?.name}
                onChange={event =>
                  setProcessor(processors?.find(({ name }) => name === event.target.value) || null)
                }
              >
                {[NO_PROCESSOR_LABEL, ...processors.map(({ name }) => name)].map(name => (
                  <option key={name} value={name}>
                    {snakeCaseToSentence(name)}
                  </option>
                ))}
              </select>
            </div>
          )}
          {processor && (
            <div>
              <div>
                <button disabled={isEstimatingParams} onClick={startEstimatingParams}>
                  Start estimating
                </button>
                <button disabled={!isEstimatingParams} onClick={stopEstimatingParams}>
                  Stop estimating
                </button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'row' }}>
                <div>
                  {processor.param_definitions.map(
                    ({ name, default_value, min_value, max_value, log_scale }) => (
                      <Slider
                        key={name}
                        name={snakeCaseToSentence(name)}
                        value={processor.params[name] || default_value || 0.0}
                        minValue={min_value}
                        maxValue={max_value}
                        logScale={log_scale}
                        onChange={newValue => {
                          const newProcessor = { ...processor }
                          newProcessor.params[name] = newValue
                          setProcessor(newProcessor)
                        }}
                      />
                    )
                  )}
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
                          />
                        )
                    )}
                  </div>
                )}
              </div>
              {trainState?.loss !== undefined && (
                <div>
                  <span>Loss: </span>
                  {trainState.loss}
                </div>
              )}
            </div>
          )}
        </>
      )}
      {isEstimatingParams && (
        <>
          {lossOptions && (
            <div>
              <span style={{ fontSize: 18, fontWeight: 'bold' }}>Loss options</span>
              <ul style={{ listStyle: 'none' }}>
                <li>
                  <label htmlFor="weights">Weights:</label>
                  <ul id="weights">
                    {Object.entries(lossOptions.weights).map(([key, value]) => (
                      <li key={key} style={{ listStyle: 'none' }}>
                        <Slider
                          key={key}
                          name={snakeCaseToSentence(key)}
                          value={value}
                          minValue={0.0}
                          maxValue={1.0}
                          onChange={newValue => {
                            const newLossOptions = { ...lossOptions }
                            newLossOptions.weights[key] = newValue
                            setLossOptions(newLossOptions)
                          }}
                        />
                      </li>
                    ))}
                  </ul>
                </li>
                <li>
                  <label htmlFor="distance_types">Distance types:</label>
                  <ul id="distance_types">
                    {Object.entries(lossOptions.distance_types).map(([key, value]) => (
                      <li id={key} key={key} style={{ listStyle: 'none' }}>
                        <label htmlFor={key}>{key}: </label>
                        <select
                          id={key}
                          value={value}
                          onChange={event => {
                            const newLossOptions = { ...lossOptions }
                            newLossOptions.distance_types[key] = event.target.value
                            setLossOptions(newLossOptions)
                          }}
                        >
                          <option value="L1">L1</option>
                          <option value="L2">L2</option>
                        </select>
                      </li>
                    ))}
                  </ul>
                </li>
              </ul>
              <button onClick={sendLossOptions}>Set loss options</button>
            </div>
          )}
          {optimizers && (
            <div>
              <span style={{ fontSize: 18, fontWeight: 'bold' }}>Optimization options</span>
              <ul style={{ listStyle: 'none' }}>
                <li>
                  <div>
                    <select
                      value={optimizer?.name}
                      onChange={event =>
                        setOptimizer(optimizers.find(({ name }) => name === event.target.value))
                      }
                    >
                      {optimizers.map(({ name }) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </div>
                </li>
                {optimizer && (
                  <li>
                    <ul>
                      {optimizer.param_definitions.map(
                        ({ name, min_value, max_value, log_scale }) =>
                          !isNaN(optimizer.params[name]) && (
                            <Slider
                              key={name}
                              name={snakeCaseToSentence(name)}
                              value={optimizer.params[name]}
                              minValue={min_value}
                              maxValue={max_value}
                              logScale={log_scale}
                              onChange={newValue => {
                                const newOptimizer = { ...optimizer }
                                newOptimizer.params[name] = newValue
                                setOptimizer(newOptimizer)
                              }}
                            />
                          )
                      )}
                    </ul>
                  </li>
                )}
              </ul>
              <button onClick={sendOptimizer}>Set optimization options</button>
            </div>
          )}
        </>
      )}
      <div>
        <button onClick={() => setIsStreamingAudio(!isStreamingAudio)}>
          {isStreamingAudio ? 'Stop sending' : 'Start sending'}
        </button>
        {audioStreamErrorMessage && (
          <div style={{ color: '#B33A3A', fontSize: '12px' }}>{audioStreamErrorMessage}.</div>
        )}
      </div>
      <audio controls autoPlay ref={audioRef} hidden></audio>
      <DraggableList direction="vertical" />
    </div>
  )
}
