import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import cv2
from aiohttp import web
import aiohttp_cors
from av import AudioFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

import numpy as np
import jax.numpy as jnp
from scipy import signal
from scipy.io.wavfile import read as readwav
import time

from jaxdsp.processors import fir_filter, iir_filter, clip, delay_line, lowpass_feedback_comb_filter as lbcf, allpass_filter, freeverb, serial_processors
from jaxdsp.training import train, evaluate, process


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

sample_rate, test_X = readwav('./audio/speech-male.wav')
processor_state = None
empty_carry = {'state': None, 'params': None}

def get_frame_ndarray(frame):
    ints = np.frombuffer(frame.planes[0], dtype=np.int16)
    floats = ints.astype(np.float32) / np.iinfo(np.int16).max
    return floats

def set_frame_ndarray(frame, floats):
    ints = (floats * np.iinfo(np.int16).max).astype(np.int16)
    frame.planes[0].update(ints)

def get_processor_and_params(processor_name):
    if processor_name == "freeverb":
        return (freeverb, {
                    'wet': 0.3,
                    'dry': 0.6,
                    'width': 0.5,
                    'damp': 0.3,
                    'room_size': 1.055,
                })
    elif processor_name == "clip":
        return (clip, {'min': -0.01, 'max': 0.01})
    elif processor_name == "delay_line":
        return (delay_line, {'wet_amount': 0.5, 'delay_samples': 99})
    else:
        return (None, None)

class AudioTransformTrack(MediaStreamTrack):
    global processor_state
    kind = "audio"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        global processor_state
        frame = await self.track.recv()
        X = get_frame_ndarray(frame)
        processor, params = get_processor_and_params(self.transform)

        carry, Y = (empty_carry, X) if processor is None else processor.tick_buffer({'params': params, 'state': processor_state or processor.init_state()}, X)
        processor_state = carry['state']
        Y = np.asarray(Y)
        if Y.ndim == 2: Y = np.sum(Y, axis=1)
        assert(Y.ndim == 1)
        set_frame_ndarray(frame, Y)
        return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            local_audio = AudioTransformTrack(
                track, transform=params["audio_transform"]
            )
            pc.addTrack(local_audio)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            expose_headers="*",
            allow_headers="*",
        )
    })
    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
