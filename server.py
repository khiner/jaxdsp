import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time
import numpy as np
import jax.numpy as jnp

from aiohttp import web
import aiohttp_cors
from av import AudioFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from jaxdsp.processors import fir_filter, iir_filter, clip, delay_line, lowpass_feedback_comb_filter, allpass_filter, freeverb

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

empty_carry = {'state': None, 'params': None}

def get_frame_ndarray(frame):
    ints = np.frombuffer(frame.planes[0], dtype=np.int16)
    floats = ints.astype(np.float32) / np.iinfo(np.int16).max
    return floats

def set_frame_ndarray(frame, floats):
    ints = (floats * np.iinfo(np.int16).max).astype(np.int16)
    frame.planes[0].update(ints)


class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"
    ALL_PROCESSORS = [clip, delay_line, lowpass_feedback_comb_filter, allpass_filter, freeverb]

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.all_estimated_params = {processor.NAME: processor.init_params() for processor in self.ALL_PROCESSORS}
        self.processor_name = 'none'
        self.processor_state = None
        self.processor_params = None
        self.processor = None
        self.processor_estimated_params = None
        self.channel = None

    def set_processor_name(self, processor_name):
        if self.processor_name != processor_name:
            self.processor_name = processor_name
            self.processor_state = None
            self.processor_params = None
            self.processor = self.get_processor()
            self.processor_estimated_params = self.all_estimated_params.get(self.processor_name)

    def set_processor_params(self, processor_params):
        self.processor_params = processor_params

    def set_channel(self, channel):
        self.channel = channel

    def get_processor(self):
        for processor in self.ALL_PROCESSORS:
            if processor.NAME == self.processor_name:
                return processor
        return None

    async def recv(self):
        frame = await self.track.recv()
        X = get_frame_ndarray(frame)
        params = self.processor_params or (self.processor and self.processor.init_params())
        carry, Y = (empty_carry, X) if self.processor is None else self.processor.tick_buffer({'params': params, 'state': self.processor_state or self.processor.init_state()}, X)
        self.processor_state = carry['state']
        Y = np.asarray(Y)
        if Y.ndim == 2: Y = np.sum(Y, axis=1) # TODO stereo out
        assert(Y.ndim == 1)
        set_frame_ndarray(frame, Y)
        if self.channel:
            self.channel.send(json.dumps({'frame': 'Processed'}))
        return frame

# RTC::track and RTC::datachannel may arrive in any order.
# This class just handles properly instantiating things regardless of received order.
class AudioTrackAndConfig():
    def __init__(self):
        self.track = None
        self.audio_processor_name = None
        self.param_values = {processor.NAME: processor.init_params() for processor in AudioTransformTrack.ALL_PROCESSORS}
        self.channel = None

    def set_track(self, track):
        self.track = track
        self.update_track()

    def set_audio_processor_name(self, audio_processor_name):
        self.audio_processor_name = audio_processor_name
        self.update_track()

    def set_param_values(self, param_values):
        self.param_values = param_values
        self.update_track()

    # While the channel is set, an 'estimated_param_values' json message is sent periodically
    def set_channel(self, channel):
        self.channel = channel
        self.update_track()

    def update_track(self):
        if self.track and self.audio_processor_name and self.param_values:
            self.track.set_processor_name(self.audio_processor_name)
            self.track.set_channel(self.channel)
            if self.audio_processor_name in self.param_values:
                self.track.set_processor_params(self.param_values[self.audio_processor_name])

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    audio_track_and_config = AudioTrackAndConfig()

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            print('Message: ', message)
            if message == 'get_config':
                channel.send(json.dumps({
                    'processors': {
                        processor.NAME: {
                            'params': [param.__dict__ for param in processor.PARAMS],
                            'presets': processor.PRESETS,
                        } for processor in AudioTransformTrack.ALL_PROCESSORS
                    },
                    'param_values': audio_track_and_config.param_values,
                }))
            elif message == 'start_estimating_params':
                audio_track_and_config.set_channel(channel)
            elif message == 'stop_estimating_params':
                audio_track_and_config.set_channel(None)
            else:
                message_object = json.loads(message)
                if 'audio_processor_name' in message_object:
                    audio_track_and_config.set_audio_processor_name(message_object['audio_processor_name'])
                if 'param_values' in message_object:
                    audio_track_and_config.set_param_values(message_object['param_values'])

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
            audio_track = AudioTransformTrack(track)
            audio_track_and_config.set_track(audio_track)
            pc.addTrack(audio_track)

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
    parser = argparse.ArgumentParser(description="JAXdsp server")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server (default: 8080)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO))

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

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
    for route in list(app.router.routes()): cors.add(route)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
