import React, { useEffect, useRef, useState } from 'react'
import { Button, Row, Select, Space, Typography } from 'antd'
import { DragDropContext } from 'react-beautiful-dnd'
import adapter from 'webrtc-adapter' // eslint-disable-line no-unused-vars

import DragDropList from './DragDropList'
import Slider from './Slider'
import Processor from './Processor'

import { negotiatePeerConnection } from '../util/WebRtc'
import { clone, deepEquals } from '../util/object'
import { snakeCaseToSentence } from '../util/string'

import 'antd/dist/antd.css'

const { Title } = Typography

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

export default function JaxDspClient({ testSample }) {
  const [audioInputSourceLabel, setAudioInputSourceLabel] = useState(AUDIO_INPUT_SOURCES.testSample.label)
  const [isStreamingAudio, setIsStreamingAudio] = useState(false)
  const [isEstimatingParams, setIsEstimatingParams] = useState(false)
  const [processorDefinitions, setProcessorDefinitions] = useState(null)
  const [optimizers, setOptimizers] = useState(null)
  const [lossOptions, setLossOptions] = useState(null)
  const [editingOptimizer, setEditingOptimizer] = useState(null)
  const [optimizer, setOptimizer] = useState(null)
  const [trainState, setTrainState] = useState({})
  const [audioStreamErrorMessage, setAudioStreamErrorMessage] = useState(null)
  const [clientUid, setClientUid] = useState(null)
  const [selectedProcessors, setSelectedProcessors] = useState([])
  const [mouseX, setMouseX] = useState(undefined)

  const audioRef = useRef(null)

  const sendProcessors = () => dataChannel?.send(JSON.stringify({ processors: selectedProcessors }))
  const sendOptimizer = () => {
    setOptimizer(editingOptimizer)
    return dataChannel?.send(JSON.stringify({ optimizer: editingOptimizer }))
  }
  const sendLossOptions = () => dataChannel?.send(JSON.stringify({ loss_options: lossOptions }))

  const onAudioStreamError = (displayMessage, error) => {
    setIsStreamingAudio(false)
    setAudioStreamErrorMessage(displayMessage)
    const errorMessage = error ? `${displayMessage}: ${error}` : displayMessage
    console.error(errorMessage)
  }

  useEffect(() => {
    const onMouseMove = event => setMouseX(event.pageX)
    window.addEventListener('mousemove', onMouseMove)
    return () => window.removeEventListener('mousemove', onMouseMove)
  }, [])

  useEffect(() => {
    sendProcessors()
  }, [selectedProcessors])

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
        const { processor_definitions, processors, optimizer_definitions, optimizer, loss_options } = message

        if (processor_definitions) setProcessorDefinitions(processor_definitions)
        if (processors) setSelectedProcessors(processors)
        if (optimizer_definitions) setOptimizers(optimizer_definitions)
        if (optimizer) {
          setEditingOptimizer(optimizer)
          setOptimizer(optimizer)
        }
        if (loss_options) setLossOptions(loss_options)
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
      setSelectedProcessors([])
      setProcessorDefinitions(null)
      setEditingOptimizer(null)
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
    <div style={{ margin: 10 }}>
      <Title level={2}>JAXdsp client</Title>
      <Title level={3} style={{ marginTop: 0 }}>Controlling remote differentiable audio graphs</Title>
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
        <div>
          {processorDefinitions && (
            <DragDropContext
              onDragEnd={({ source, destination }) => {
                if (destination) {
                  if (
                    source.droppableId === 'selectedProcessors' &&
                    destination.droppableId === 'selectedProcessors'
                  ) {
                    const reorderedProcessors = [...selectedProcessors]
                    const [removed] = reorderedProcessors.splice(source.index, 1)
                    reorderedProcessors.splice(destination.index, 0, removed)
                    setSelectedProcessors(reorderedProcessors)
                  } else if (
                    source.droppableId === 'processors' &&
                    destination.droppableId === 'selectedProcessors'
                  ) {
                    const item = clone(processorDefinitions[source.index])
                    const newSelectedProcessors = [...selectedProcessors]
                    newSelectedProcessors.splice(destination.index, 0, item)
                    setSelectedProcessors(newSelectedProcessors)
                  }
                }
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <div
                  style={{
                    width: 'fit-content',
                    display: 'flex',
                    flexDirection: 'row',
                    background: '#e0e0e0',
                    borderRadius: '4px',
                    margin: '4px',
                    alignItems: 'center',
                  }}
                >
                  <label style={{ fontSize: '17px', fontWeight: 'bold', margin: '0px 8px' }}>
                    Processors
                  </label>
                  <DragDropList
                    itemDraggingStyle={{ background: 'white' }}
                    droppableId="processors"
                    direction="horizontal"
                    isStatic
                  >
                    {processorDefinitions.map(({ name }) => (
                      <div key={name}>{name}</div>
                    ))}
                  </DragDropList>
                </div>
                <DragDropList
                  style={{
                    height: 'fit-content',
                    background: '#e0e0e0',
                    borderRadius: '4px',
                    width: 'fit-content',
                  }}
                  draggingStyle={{ outline: '1px dashed black', background: '#e0e0e0' }}
                  droppableId="selectedProcessors"
                  direction="horizontal"
                  emptyContent={<i style={{ margin: '8px' }}>Drop processors here</i>}
                >
                  {selectedProcessors.map((processor, i) => (
                    <Processor
                      key={i}
                      processor={processor}
                      estimatedParams={trainState?.['params']?.[i]}
                      mouseX={mouseX}
                      onChange={(paramName, newValue) => {
                        const newSelectedProcessors = clone(selectedProcessors)
                        newSelectedProcessors[i].params[paramName] = newValue
                        setSelectedProcessors(newSelectedProcessors)
                      }}
                      onClose={() => {
                        const newSelectedProcessors = clone(selectedProcessors)
                        newSelectedProcessors.splice(i, 1)
                        setSelectedProcessors(newSelectedProcessors)
                      }}
                    />
                  ))}
                </DragDropList>
              </div>
            </DragDropContext>
          )}
          {isEstimatingParams && trainState?.loss !== undefined && (
            <div>
              <span>Loss: </span>
              {trainState.loss}
            </div>
          )}
        </div>
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
                          mouseX={mouseX}
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
                              mouseX={mouseX}
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
