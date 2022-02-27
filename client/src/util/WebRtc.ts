// Adapted from https://github.com/aiortc/aiortc/blob/main/examples/server/client.js#L212-L271
export const sdpFilterCodec = (kind, codec, realSdp) => {
  const escapeRegExp = string => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') // $& means the whole matched string

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

export const negotiatePeerConnection = async (peerConnection, offerUrl): Promise<void> => {
  const offer = await peerConnection.createOffer()
  // TODO somewhere in the chain from here to the server,
  // the stream is getting mixed to mono and split back into identical
  // L/R channels. It seems an awful lot like
  // [this bug](https://bugs.chromium.org/p/webrtc/issues/detail?id=8133),
  // but that bug was resolved in Chrome v89 (currently v88 - Feb 4),
  // and also it focuses on microphone sources.
  // I notice that the answer sdp doesn't even have the fmtp line...
  offer.sdp = offer.sdp.replace('a=fmtp:111', 'a=fmtp:111 stereo=1;sprop-stereo=1;')
  await peerConnection.setLocalDescription(offer)
  await new Promise<void>(resolve => {
    if (peerConnection.iceGatheringState === 'complete') {
      resolve()
    } else {
      const checkState = () => {
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

  // Currently 20ms at 44kHz results in 960-sample packets (per-channel).
  // This limits the max power-of-2 fft size to 512, which limits the lowest resolved frequency to 44100 / 512 = 87Hz
  // It would be nice (maybe even crucial?) to increase the packet size from 20ms to 40ms, to allow for >= 1024 sample fft sizes.
  // However, changing the sdp directly here doesn't seem to change anything.
  // Also, [media track constraints](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints)
  // don't offer anything related to packet size.
  // My workaround is to just process P > 1 packets at a time on the server side,
  // introducing P - 1 packets of processing delay. (See `server.py::AudioTransformTrack::__init__` for more details.)
  // peerConnection.localDescription.sdp = peerConnection.localDescription.sdp.replace(/minptime=\d+/, 'minptime=40')
  // peerConnection.localDescription.sdp += 'a=ptime:40\n';
  const response = await fetch(offerUrl, {
    body: JSON.stringify({
      sdp: peerConnection.localDescription.sdp,
      type: peerConnection.localDescription.type,
    }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  })
  const answer = await response.json()
  const { client_uid: clientUid, ...remoteDescription } = answer
  await peerConnection.setRemoteDescription(remoteDescription)

  return clientUid
}
