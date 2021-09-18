import React, { useEffect, useRef, useState } from 'react'
import { Button, Row, Select, Space, Typography } from 'antd'
import adapter from 'webrtc-adapter' // eslint-disable-line no-unused-vars

import Slider from './Slider'

import { negotiatePeerConnection } from '../util/WebRtc'
import { clone, deepEquals } from '../util/object'
import { snakeCaseToSentence } from '../util/string'

import 'antd/dist/antd.css'
import RealtimeController from './RealtimeController'

const { Title } = Typography

const serverUrl = 'http://localhost:8080'

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

const get = async (path, clientUid) => {
  const response = await fetch(`${serverUrl}/${path}/${clientUid}`, {
    headers: { 'Content-Type': 'application/json' },
    method: 'GET',
  })
  return response.json()
}

const post = async (path, clientUid, postBody = undefined) =>
  await fetch(`${serverUrl}/${path}/${clientUid}`, {
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    ...(postBody ? { body: JSON.stringify(postBody) } : {}),
  })

export default function JaxDspClient({ testSample }) {
  const [audioInputSourceLabel, setAudioInputSourceLabel] = useState(AUDIO_INPUT_SOURCES.testSample.label)
  const [isStreamingAudio, setIsStreamingAudio] = useState(false)
  const [isEstimatingParams, setIsEstimatingParams] = useState(false)
  const [processorDefinitions, setProcessorDefinitions] = useState(null)
  const [optimizers, setOptimizers] = useState(null)
  const [lossOptions, setLossOptions] = useState(null)
  const [editingOptimizer, setEditingOptimizer] = useState(null)
  const [optimizer, setOptimizer] = useState(null)
  const [audioStreamErrorMessage, setAudioStreamErrorMessage] = useState(null)
  const [clientUid, setClientUid] = useState(null)
  const [selectedProcessors, setSelectedProcessors] = useState([])

  const audioRef = useRef(null)

  const sendProcessors = async () => {
    await post('state', clientUid, { processors: selectedProcessors })
  }
  const sendOptimizer = async () => {
    setOptimizer(editingOptimizer)
    await post('state', clientUid, { optimizer: editingOptimizer })
  }
  const sendLossOptions = async () => {
    await post('state', clientUid, { loss_options: lossOptions })
  }

  const onAudioStreamError = (displayMessage, error) => {
    setIsStreamingAudio(false)
    setAudioStreamErrorMessage(displayMessage)
    const errorMessage = error ? `${displayMessage}: ${error}` : displayMessage
    console.error(errorMessage)
  }

  const getState = async () => {
    const response = await get('state', clientUid)
    const { processor_definitions, processors, optimizer_definitions, optimizer, loss_options } = response

    if (processor_definitions) setProcessorDefinitions(processor_definitions)
    if (processors) setSelectedProcessors(processors)
    if (optimizer_definitions) setOptimizers(optimizer_definitions)
    if (optimizer) {
      setEditingOptimizer(optimizer)
      setOptimizer(optimizer)
    }
    if (loss_options) setLossOptions(loss_options)
  }

  useEffect(() => {
    if (clientUid) getState()
  }, [clientUid])

  useEffect(() => {
    if (clientUid) sendProcessors()
  }, [selectedProcessors])

  useEffect(() => {
    const openPeerConnection = () => {
      if (peerConnection) return

      peerConnection = new RTCPeerConnection()
      peerConnection.onconnectionstatechange = () => {
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
      peerConnection.addEventListener('track', async event => {
        audioRef.current.srcObject = event.streams[0]
      })
      dataChannel = peerConnection.createDataChannel('jaxdsp-client', { ordered: true })
    }

    const closePeerConnection = () => {
      if (!peerConnection) return

      peerConnection.getSenders().forEach(sender => sender?.track?.stop())
      dataChannel?.close()
      dataChannel = null
      peerConnection.getTransceivers()?.forEach(transceiver => transceiver.stop())
      peerConnection.close()
      peerConnection = null
      setSelectedProcessors([])
      setProcessorDefinitions(null)
      setEditingOptimizer(null)
      setOptimizers(null)
      setIsEstimatingParams(false)
      setClientUid(null)
    }

    const addOrReplaceTrack = async track => {
      const audioSender = peerConnection.getSenders().find(s => s.track?.kind === 'audio')
      if (audioSender) {
        await audioSender.replaceTrack(track)
      } else {
        peerConnection.addTrack(track)
        try {
          const clientUid = await negotiatePeerConnection(peerConnection, `${serverUrl}/offer`)
          setClientUid(clientUid)
        } catch (error) {
          onAudioStreamError('Failed to negotiate RTC peer connection', error)
        }
      }
    }

    const startStreamingMicrophone = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false, channelCount: 2 },
          video: false,
        })
        await addOrReplaceTrack(stream.getTracks()[0])
      } catch (error) {
        onAudioStreamError('Could not acquire media', error)
      }
    }

    const startStreamingTestSample = async () => {
      const sampleAudio = new Audio(testSample)
      const audioContext = new AudioContext()
      const sampleSource = audioContext.createMediaElementSource(sampleAudio)
      const sampleDestination = audioContext.createMediaStreamDestination()
      sampleSource.connect(sampleDestination)
      await addOrReplaceTrack(sampleDestination.stream.getAudioTracks()[0])

      sampleAudio.loop = true
      sampleAudio.currentTime = 0
      await sampleAudio.play()
    }

    if (isStreamingAudio) {
      setAudioStreamErrorMessage(null)
      openPeerConnection()
      if (audioInputSourceLabel === AUDIO_INPUT_SOURCES.microphone.label) startStreamingMicrophone()
      else if (audioInputSourceLabel === AUDIO_INPUT_SOURCES.testSample.label) startStreamingTestSample()
    } else if (!isStreamingAudio) {
      closePeerConnection()
    }

    return () => closePeerConnection()
  }, [isStreamingAudio, audioInputSourceLabel])

  const startEstimatingParams = async () => {
    await post('start_estimating', clientUid)
    setIsEstimatingParams(true)
  }

  const stopEstimatingParams = async () => {
    await post('stop_estimating_params', clientUid)
    setIsEstimatingParams(false)
  }

  return (
    <div style={{ margin: 10 }}>
      <Title level={2}>JAXdsp client</Title>
      <Title level={3} style={{ marginTop: 0 }}>
        Real-time remote control and training of differentiable audio graphs
      </Title>
      <div>
        <span>Audio input source:</span>{' '}
        <Select value={audioInputSourceLabel} onChange={value => setAudioInputSourceLabel(value)}>
          {Object.values(AUDIO_INPUT_SOURCES).map(({ label }) => (
            <Select.Option key={label} value={label}>
              {label}
            </Select.Option>
          ))}
        </Select>
      </div>
      {isStreamingAudio && (
        <RealtimeController
          clientUid={clientUid}
          processorDefinitions={processorDefinitions}
          selectedProcessors={selectedProcessors}
          setSelectedProcessors={setSelectedProcessors}
          onError={setAudioStreamErrorMessage}
        />
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
                        <Select
                          id={key}
                          value={value}
                          onChange={value => {
                            const newLossOptions = { ...lossOptions }
                            newLossOptions.distance_types[key] = value
                            setLossOptions(newLossOptions)
                          }}
                        >
                          <Select.Option value="L1">L1</Select.Option>
                          <Select.Option value="L2">L2</Select.Option>
                        </Select>
                      </li>
                    ))}
                  </ul>
                </li>
              </ul>
              <Button onClick={sendLossOptions}>Set loss options</Button>
            </div>
          )}
          {optimizers && (
            <div>
              <span style={{ fontSize: 18, fontWeight: 'bold' }}>Optimization options</span>
              <ul style={{ listStyle: 'none' }}>
                <li>
                  <div>
                    <Select
                      value={editingOptimizer?.name}
                      onChange={value => setEditingOptimizer(optimizers.find(({ name }) => name === value))}
                    >
                      {optimizers.map(({ name }) => (
                        <Select.Option key={name} value={name}>
                          {name}
                        </Select.Option>
                      ))}
                    </Select>
                  </div>
                </li>
                {editingOptimizer && (
                  <li>
                    <ul>
                      {editingOptimizer.param_definitions.map(
                        ({ name, min_value, max_value, log_scale }) =>
                          !isNaN(editingOptimizer.params[name]) && (
                            <Slider
                              key={name}
                              name={snakeCaseToSentence(name)}
                              value={editingOptimizer.params[name]}
                              minValue={min_value}
                              maxValue={max_value}
                              logScale={log_scale}
                              onChange={newValue => {
                                const newOptimizer = clone(editingOptimizer)
                                newOptimizer.params[name] = newValue
                                setEditingOptimizer(newOptimizer)
                              }}
                            />
                          )
                      )}
                    </ul>
                  </li>
                )}
              </ul>
              <Button disabled={deepEquals(optimizer, editingOptimizer)} onClick={sendOptimizer}>
                Set optimization options
              </Button>
            </div>
          )}
        </>
      )}
      <div>
        <Row style={{ margin: '5px 0' }}>
          <Space>
            <Button type="primary" onClick={() => setIsStreamingAudio(!isStreamingAudio)}>
              {isStreamingAudio ? 'Stop streaming audio' : 'Start streaming audio'}
            </Button>
            {isStreamingAudio && (
              <Button
                disabled={!isEstimatingParams && !selectedProcessors?.length}
                onClick={isEstimatingParams ? stopEstimatingParams : startEstimatingParams}
              >
                {`${isEstimatingParams ? 'Stop' : 'Start'} estimating params`}
              </Button>
            )}
          </Space>
        </Row>
        {audioStreamErrorMessage && (
          <div style={{ color: '#B33A3A', fontSize: '12px' }}>{audioStreamErrorMessage}.</div>
        )}
      </div>
      <audio controls autoPlay ref={audioRef} hidden></audio>
    </div>
  )
}
