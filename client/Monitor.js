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
const NONE_PROCESSOR_LABEL = 'None'

export default function Monitor({ testSample }) {
  const [audioInputSourceLabel, setAudioInputSourceLabel] = useState(AUDIO_INPUT_SOURCES.testSample.label)
  const [isStreamingAudio, setIsStreamingAudio] = useState(false)
  const [isEstimatingParams, setIsEstimatingParams] = useState(false)
  const [processorName, setProcessorName] = useState(NONE_PROCESSOR_LABEL)
  const [processors, setProcessors] = useState(null)
  const [paramValues, setParamValues] = useState({})
  const [trainState, setTrainState] = useState({})
  const [audioStreamErrorMessage, setAudioStreamErrorMessage] = useState(null)
  const [clientUid, setClientUid] = useState(null)

  const onAudioStreamError = (displayMessage, error) => {
    setIsStreamingAudio(false)
    setAudioStreamErrorMessage(displayMessage)
    const errorMessage = error ? `${displayMessage}: ${error}` : displayMessage
    console.error(errorMessage)
  }

  const updateParamValue = (paramName, value) => {
    const newParamValues = { ...paramValues }
    if (newParamValues[processorName]) {
      newParamValues[processorName][paramName] = value
    }
    setParamValues(newParamValues)
  }

  const audioRef = useRef(null)

  useEffect(() => {
    dataChannel?.send(JSON.stringify({ audio_processor_name: processorName }))
  }, [processorName])

  useEffect(() => {
    dataChannel?.send(JSON.stringify({ param_values: paramValues }))
  }, [paramValues])

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
      dataChannel.onopen = () => dataChannel.send('get_config')
      dataChannel.onmessage = event => {
        const message = JSON.parse(event.data)
        const { processors, param_values: paramValues } = message

        if (processors) {
          setProcessors(processors)
        }
        if (paramValues) {
          setParamValues(paramValues)
        }
      }
    }

    const closePeerConnection = () => {
      if (!peerConnection) return

      peerConnection.getSenders().forEach(sender => sender?.track?.stop())
      dataChannel?.close()
      peerConnection.getTransceivers()?.forEach(transceiver => transceiver.stop())
      peerConnection.close()
      peerConnection = null
    }

    const addOrReplaceTrack = async track => {
      const audioSender = peerConnection.getSenders().find(s => s.track && s.track.kind === 'audio')
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
      const testSampleAudio = new Audio(testSample)
      const audioContext = new AudioContext()
      const testSampleSource = audioContext.createMediaElementSource(testSampleAudio)
      const testSampleDestination = audioContext.createMediaStreamDestination()
      testSampleSource.connect(testSampleDestination)

      const testSampleTrack = testSampleDestination.stream.getAudioTracks()[0]
      addOrReplaceTrack(testSampleTrack)

      testSampleAudio.loop = true
      testSampleAudio.currentTime = 0
      testSampleAudio.play()
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

  const processorParams = processors && processors[processorName] && processors[processorName].params
  const processorParamValues = (paramValues && paramValues[processorName]) || {}
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
      {processors && (
        <div>
          <select value={processorName} onChange={event => setProcessorName(event.target.value)}>
            {[NONE_PROCESSOR_LABEL, ...Object.keys(processors)].map(processorName => (
              <option key={processorName} value={processorName}>
                {processorName}
              </option>
            ))}
          </select>
        </div>
      )}
      {processorParams && (
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
              {processorParams.map(({ name, default_value, min_value, max_value }) => (
                <div key={name}>
                  <input
                    type="range"
                    name={name}
                    value={processorParamValues[name] || default_value || 0.0}
                    min={min_value}
                    max={max_value}
                    step={(max_value - min_value) / 100.0}
                    onChange={event => updateParamValue(name, +event.target.value)}
                  />
                  <label htmlFor={name}>{name}</label>
                </div>
              ))}
            </div>
            {trainState && trainState.params && (
              <div>
                {processorParams.map(
                  ({ name, min_value, max_value }) =>
                    !isNaN(trainState.params[name]) && (
                      <div key={name}>
                        <input
                          type="range"
                          name={name}
                          value={trainState.params[name]}
                          min={min_value}
                          max={max_value}
                          step={(max_value - min_value) / 100.0}
                          onChange={event => updateParamValue(name, +event.target.value)}
                          disabled
                        />
                        <label htmlFor={name}>{name}</label>
                      </div>
                    )
                )}
              </div>
            )}
            {trainState && trainState.loss !== undefined && <div>{trainState.loss}</div>}
          </div>
        </div>
      )}
      <div>
        <button disabled={isStreamingAudio} onClick={() => setIsStreamingAudio(true)}>
          Start sending
        </button>
        <button disabled={!isStreamingAudio} onClick={() => setIsStreamingAudio(false)}>
          Stop sending
        </button>
        {audioStreamErrorMessage && (
          <div style={{ color: '#B33A3A', fontSize: '12px' }}>{audioStreamErrorMessage}.</div>
        )}
      </div>
      <audio controls autoPlay ref={audioRef} hidden></audio>
    </div>
  )
}
