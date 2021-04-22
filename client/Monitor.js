import React, { useEffect, useRef, useState } from 'react'
import adapter from 'webrtc-adapter' // eslint-disable-line no-unused-vars

// Starting point for this code from:
// https://webrtc.github.io/samples/src/content/peerconnection/webaudio-input/

let peerConnection = null
let dataChannel = null

// Adapted from https://github.com/aiortc/aiortc/blob/main/examples/server/client.js#L212-L271
function sdpFilterCodec(kind, codec, realSdp) {
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') // $& means the whole matched string
  }

  const rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$')
  const codecRegex = new RegExp(`a=rtpmap:([0-9]+) ${escapeRegExp(codec)}`)

  const lines = realSdp.split('\n')

  let isKind = false
  const allowed = lines
    .map(line => {
      if (line.startsWith(`m=${kind} `)) {
        isKind = true
      } else if (line.startsWith('m=')) {
        isKind = false
      }

      if (!isKind) return null

      let match = line.match(codecRegex)
      if (match) return parseInt(match[1])

      match = line.match(rtxRegex)
      if (match && allowed.includes(parseInt(match[2]))) return parseInt(match[1])

      return null
    })
    .filter(match => !!match)

  const skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)'
  let sdp = ''

  isKind = false
  lines.forEach(line => {
    if (line.startsWith(`m=${kind} `)) {
      isKind = true
    } else if (line.startsWith('m=')) {
      isKind = false
    }

    const skipMatch = line.match(skipRegex)
    if ((isKind && !(skipMatch && !allowed.includes(parseInt(skipMatch[2])))) || !isKind) {
      sdp += `${line}\n`
    }
  })

  return sdp
}

async function negotiate() {
  const offer = await peerConnection.createOffer()
  await peerConnection.setLocalDescription(offer)
  await new Promise(resolve => {
    if (peerConnection.iceGatheringState === 'complete') {
      resolve()
    } else {
      function checkState() {
        if (peerConnection.iceGatheringState === 'complete') {
          peerConnection.removeEventListener('icegatheringstatechange', checkState)
          resolve()
        }
      }
      peerConnection.addEventListener('icegatheringstatechange', checkState)
    }
  })
  peerConnection.localDescription.sdp = sdpFilterCodec(
    'audio',
    'opus/48000/2',
    peerConnection.localDescription.sdp
  )
  const response = await fetch('http://localhost:8080/offer', {
    body: JSON.stringify({
      sdp: peerConnection.localDescription.sdp,
      type: peerConnection.localDescription.type,
    }),
    headers: {
      'Content-Type': 'application/json',
    },
    method: 'POST',
  })
  const answer = await response.json()
  const { client_uid: clientUid, ...remoteDescription } = answer
  await peerConnection.setRemoteDescription(remoteDescription)

  return clientUid
}

const AUDIO_INPUT_SOURCES = {
  microphone: {
    label: 'Microphone',
  },
  testSample: {
    label: 'Test sample',
  },
}

const NO_PROCESSOR_LABEL = 'None'

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

export default function Monitor({ testSample }) {
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
        const { processors, processor, optimizers, optimizer, loss_options: lossOptions } = message

        if (processors) setProcessors(processors)
        if (processor) setProcessor(processor)
        if (optimizers) setOptimizers(optimizers)
        if (optimizer) setOptimizer(optimizer)
        if (lossOptions) setLossOptions(lossOptions)
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
          const clientUid = await negotiate()
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
        const microphoneTrack = stream.getTracks()[0]
        addOrReplaceTrack(microphoneTrack)
      } catch (error) {
        onAudioStreamError('Could not acquire media', error)
      }
    }

    const startStreamingTestSample = () => {
      const sampleAudio = new Audio(testSample)
      const audioContext = new AudioContext()
      const sampleSource = audioContext.createMediaElementSource(sampleAudio)
      const sampleDestination = audioContext.createMediaStreamDestination()
      // TODO somewhere in the chain from here to the server,
      // the stream is getting mixed to mono and split back into identical
      // L/R channels. It seems an awful lot like
      // [this bug](https://bugs.chromium.org/p/webrtc/issues/detail?id=8133),
      // which should be resolved in Chrome v89 (currently v88 - Feb 4).
      // ALSO, the returned track seems to be downmixed from an interleaved
      // stereo channel back to mono... so maybe it's a transport/sdp thing?
      sampleSource.connect(sampleDestination)

      const sampleTrack = sampleDestination.stream.getAudioTracks()[0]
      addOrReplaceTrack(sampleTrack)

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
                    {name}
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
                        name={name}
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
                            name={name}
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
                          name={key}
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
                              name={name}
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
    </div>
  )
}
