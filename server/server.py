import argparse
import asyncio
import websockets
import uuid
import json
import logging
import os
import ssl
import uuid
import time
import numpy as np
import jax.numpy as jnp
from collections import deque

from aiohttp import web
import aiohttp_cors
from av import AudioFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from jaxdsp.processors import fir_filter, iir_filter, clip, delay_line, lowpass_feedback_comb_filter, allpass_filter, freeverb
from jaxdsp.training import IterativeTrainer

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

ALL_PROCESSORS = [clip, delay_line,
                  lowpass_feedback_comb_filter, allpass_filter, freeverb]
EMPTY_CARRY = {'state': None, 'params': None}

TRAIN_STACK_MAX_SIZE = 100
track_for_client_uid = {}


class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track, train_stack):
        super().__init__()
        self.track = track
        self.train_stack = train_stack
        self.processor = None
        self.processor_name = 'none'
        self.processor_state = None
        self.processor_params = None
        self.is_estimating_params = False
        self.websocket = None

    def set_processor_name(self, processor_name):
        if self.processor_name != processor_name:
            self.processor_name = processor_name
            self.processor_state = None
            self.processor_params = None
            self.processor = self.get_processor()

    def set_processor_params(self, processor_params):
        self.processor_params = processor_params

    def get_processor(self):
        for processor in ALL_PROCESSORS:
            if processor.NAME == self.processor_name:
                return processor
        return None

    async def recv(self):
        frame = await self.track.recv()
        num_channels = len(frame.layout.channels)
        assert frame.format.is_packed, f'Processing assumes frames are packed, but frame is planar'
        assert num_channels == 2, f'Processing assumes frames have 2 channels'

        X_interleaved = np.frombuffer(frame.planes[0], dtype=np.int16).astype(
            np.float32) / np.iinfo(np.int16).max
        X_deinterleaved = [X_interleaved[channel_num::num_channels]
                           for channel_num in range(num_channels)]
        X_left = X_deinterleaved[0]  # TODO handle stereo in
        params = self.processor_params or (
            self.processor and self.processor.init_params())
        carry, Y_deinterleaved = (EMPTY_CARRY, X_left) if self.processor is None else self.processor.tick_buffer(
            {'params': params, 'state': self.processor_state or self.processor.init_state()}, X_left)
        self.processor_state = carry['state']
        Y_deinterleaved = np.asarray(Y_deinterleaved)
        if Y_deinterleaved.ndim == 1:
            Y_deinterleaved = np.array([Y_deinterleaved, Y_deinterleaved])
        else:
            assert(Y_deinterleaved.ndim == 2)
            # Transposing to conform to processors with stereo output.
            # Stereo processing is done that way to support the same
            # array-of-ticks processing for both stereo and mono.
            Y_deinterleaved = Y_deinterleaved.T

        Y_interleaved = np.empty(
            (Y_deinterleaved.shape[1] * 2,), dtype=Y_deinterleaved.dtype)
        Y_interleaved[0::2] = Y_deinterleaved[0]
        Y_interleaved[1::2] = Y_deinterleaved[1]
        frame.planes[0].update(
            (Y_interleaved * np.iinfo(np.int16).max).astype(np.int16))
        if self.is_estimating_params:
            self.train_stack.append([X_deinterleaved, Y_deinterleaved])
        return frame


class AudioTrackAndConfig():
    """
    RTC::track and RTC::datachannel may arrive in any order.
    This class just handles properly instantiating things regardless of received order.
    """

    def __init__(self):
        self.track = None
        self.audio_processor_name = None
        self.is_estimating_params = False
        self.param_values = {processor.NAME: processor.init_params()
                             for processor in ALL_PROCESSORS}

    def set_track(self, track):
        self.track = track
        self.update_track()

    def set_audio_processor_name(self, audio_processor_name):
        self.audio_processor_name = audio_processor_name
        self.update_track()

    def set_param_values(self, param_values):
        self.param_values = param_values
        self.update_track()

    def start_estimating_params(self):
        self.track.is_estimating_params = True

    def stop_estimating_params(self):
        self.track.is_estimating_params = False

    def update_track(self):
        if self.track and self.audio_processor_name and self.param_values:
            self.track.set_processor_name(self.audio_processor_name)
            if self.audio_processor_name in self.param_values:
                self.track.set_processor_params(
                    self.param_values[self.audio_processor_name])


async def index(request):
    return web.Response(content_type="text/plain", text="Use the `/offer` endpoint to negotiate a WebRTC peer connection.")


async def offer(request):
    client_uid = str(uuid.uuid4())
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
            if message == 'get_config':
                channel.send(json.dumps({
                    'processors': {
                        processor.NAME: {
                            'params': [param.__dict__ for param in processor.PARAMS],
                            'presets': processor.PRESETS,
                        } for processor in ALL_PROCESSORS
                    },
                    'param_values': audio_track_and_config.param_values,
                }))
            elif message == 'start_estimating_params':
                audio_track_and_config.start_estimating_params()
            elif message == 'stop_estimating_params':
                audio_track_and_config.stop_estimating_params()
            else:
                message_object = json.loads(message)
                if 'audio_processor_name' in message_object:
                    audio_track_and_config.set_audio_processor_name(
                        message_object['audio_processor_name'])
                if 'param_values' in message_object:
                    audio_track_and_config.set_param_values(
                        message_object['param_values'])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        log_info("Track %s received", track.kind)
        train_stack = deque([], TRAIN_STACK_MAX_SIZE)
        audio_track = AudioTransformTrack(track, train_stack)
        track_for_client_uid[client_uid] = audio_track
        audio_track_and_config.set_track(audio_track)
        pc.addTrack(audio_track)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            del track_for_client_uid[client_uid]

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "client_uid": client_uid
            }
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def register_websocket(websocket, path):
    client_uid = None
    while client_uid == None:
        message = await websocket.recv()
        message_object = json.loads(message)
        client_uid = message_object.get('client_uid')

    track = track_for_client_uid.get(client_uid)
    if not track:
        print(f'No track cached for client_uid {client_uid}')

    trainer_for_processor = {processor.NAME: IterativeTrainer(
        processor) for processor in ALL_PROCESSORS}
    train_stack = track.train_stack
    while True:
        try:
            if len(train_stack) > 0:
                train_pair = train_stack.pop()
                trainer = trainer_for_processor[track.processor_name]
                X, Y = train_pair
                X_left = X[0]  # TODO support stereo in
                trainer.step(X_left, Y)
                await websocket.send(json.dumps({'train_state': trainer.params_and_loss()}))
            await asyncio.sleep(0.01)
        except websockets.ConnectionClosed:
            print('ws terminated')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAXdsp server")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for HTTP server (default: 8080)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO))

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    start_server = websockets.serve(
        register_websocket, "localhost", 8765, ssl=ssl_context)
    asyncio.get_event_loop().run_until_complete(start_server)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
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
