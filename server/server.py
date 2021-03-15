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
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from jaxdsp.processors import (
    allpass_filter,
    clip,
    lowpass_feedback_comb_filter,
    sine_wave,
    processor_by_name,
    empty_carry,
)
from jaxdsp.training import IterativeTrainer
from jaxdsp.loss import LossOptions

ALL_PROCESSORS = [allpass_filter, clip, lowpass_feedback_comb_filter, sine_wave]
DEFAULT_PARAM_VALUES = {
    processor.NAME: processor.config().params_init for processor in ALL_PROCESSORS
}
# Training frame pairs are queued up for each client, limited to this cap:
MAX_TRAIN_FRAMES_PER_CLIENT = 100

logger = logging.getLogger("pc")
track_for_client_uid = {}
peer_connections = set()


class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track, train_stack):
        super().__init__()
        self.track = track
        self.train_stack = train_stack
        # Keep a memory of param values for each processor, so that clients can switch between
        # processors without losing the params they set.
        # This param_values object is updated wholesale by the client.
        # Note that processor state is reset to initial conditions when the processor changes,
        # since frames may have been processed by a different processor between switching away and back
        # to a processor. (See `set_processor`.)
        self.param_values = DEFAULT_PARAM_VALUES
        self.processor = None
        self.processor_state = None
        self.loss_options = None
        self.is_estimating_params = False

    def set_processor(self, processor):
        if self.processor and processor and self.processor.NAME == processor.NAME:
            return

        self.processor = processor
        self.processor_state = processor.config().state_init if processor else None
        self.trainer = (
            IterativeTrainer(processor, self.loss_options) if processor else None
        )

    def set_loss_options(self, loss_options):
        self.loss_options = loss_options
        if self.trainer:
            self.trainer.set_loss_options(loss_options)

    def start_estimating_params(self):
        self.is_estimating_params = True

    def stop_estimating_params(self):
        self.is_estimating_params = False

    async def recv(self):
        assert self.track
        frame = await self.track.recv()
        num_channels = len(frame.layout.channels)
        assert (
            frame.format.is_packed
        ), "Processing assumes frames are packed, but frame is planar"
        assert num_channels == 2, "Processing assumes frames have 2 channels"

        X_interleaved = (
            np.frombuffer(frame.planes[0], dtype=np.int16).astype(np.float32)
            / np.iinfo(np.int16).max
        )
        X_deinterleaved = [
            X_interleaved[channel_num::num_channels]
            for channel_num in range(num_channels)
        ]
        X_left = X_deinterleaved[0]  # TODO handle stereo in
        if self.processor:
            self.processor_state["sample_rate"] = frame.sample_rate
            carry, Y_deinterleaved = self.processor.tick_buffer(
                {
                    "params": self.param_values[self.processor.NAME],
                    "state": self.processor_state,
                },
                X_left,
            )
        else:
            carry, Y_deinterleaved = (empty_carry, X_left)

        self.processor_state = carry["state"]
        Y_deinterleaved = np.asarray(Y_deinterleaved)
        if Y_deinterleaved.ndim == 1:
            Y_deinterleaved = np.array([Y_deinterleaved, Y_deinterleaved])
        else:
            assert Y_deinterleaved.ndim == 2
            # Transposing to conform to processors with stereo output.
            # Stereo processing is done that way to support the same
            # array-of-ticks processing for both stereo and mono.
            Y_deinterleaved = Y_deinterleaved.T

        Y_interleaved = np.empty(
            (Y_deinterleaved.shape[1] * 2,), dtype=Y_deinterleaved.dtype
        )
        Y_interleaved[0::2] = Y_deinterleaved[0]
        Y_interleaved[1::2] = Y_deinterleaved[1]
        frame.planes[0].update(
            (Y_interleaved * np.iinfo(np.int16).max).astype(np.int16)
        )
        if self.is_estimating_params:
            self.train_stack.append([X_deinterleaved, Y_deinterleaved])

        return frame


async def index(request):
    return web.Response(
        content_type="text/plain",
        text="Use the `/offer` endpoint to negotiate a WebRTC peer connection.",
    )


async def offer(request):
    client_uid = str(uuid.uuid4())
    audio_transform_track = AudioTransformTrack(
        None, deque([], MAX_TRAIN_FRAMES_PER_CLIENT)
    )

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    peer_connection = RTCPeerConnection()
    peer_connections.add(peer_connection)

    print(f"pcs: {len(peer_connections)}")
    peer_connection_id = "PeerConnection(%s)" % uuid.uuid4()

    def log_info(msg, *args):
        logger.info(peer_connection_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @peer_connection.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if message == "get_processors":
                channel.send(
                    json.dumps(
                        {
                            "processors": {
                                processor.NAME: {
                                    "params": [
                                        param.serialize() for param in processor.PARAMS
                                    ],
                                    "presets": processor.PRESETS,
                                }
                                for processor in ALL_PROCESSORS
                            },
                        }
                    )
                )
            elif message == "get_param_values":
                channel.send(
                    json.dumps({"param_values": audio_transform_track.param_values})
                )
            elif message == "start_estimating_params":
                audio_transform_track.start_estimating_params()
            elif message == "stop_estimating_params":
                audio_transform_track.stop_estimating_params()
            else:
                message_dict = json.loads(message)
                if "audio_processor_name" in message_dict:
                    audio_transform_track.set_processor(
                        processor_by_name.get(message_dict["audio_processor_name"])
                    )
                if "param_values" in message_dict:
                    audio_transform_track.param_values = message_dict["param_values"]
                if "loss_options" in message_dict:
                    loss_options = message_dict["loss_options"]
                    audio_transform_track.set_loss_options(
                        LossOptions(
                            weights=loss_options["weights"],
                            distance_types=loss_options["distance_types"],
                            fft_sizes=loss_options["fft_sizes"],
                        )
                    )

    @peer_connection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = peer_connection.iceConnectionState
        log_info("ICE connection state is %s", state)
        if state == "failed" or state == "closed":
            await peer_connection.close()
            peer_connections.discard(peer_connection)

    @peer_connection.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        log_info("Track %s received", track.kind)

        audio_transform_track.track = track
        track_for_client_uid[client_uid] = audio_transform_track
        peer_connection.addTrack(audio_transform_track)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            del track_for_client_uid[client_uid]

    # handle offer
    await peer_connection.setRemoteDescription(offer)

    # send answer
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": peer_connection.localDescription.sdp,
                "type": peer_connection.localDescription.type,
                "client_uid": client_uid,
            }
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()


async def register_websocket(websocket, path):
    client_uid = None
    while client_uid == None:
        message = await websocket.recv()
        message_object = json.loads(message)
        client_uid = message_object.get("client_uid")

    track = track_for_client_uid.get(client_uid)
    if not track:
        print(f"No track cached for client_uid {client_uid}")

    train_stack = track.train_stack
    while True:
        try:
            if track.is_estimating_params and track.trainer and len(train_stack) > 0:
                train_pair = train_stack.pop()
                X, Y = train_pair
                X_left = X[0]  # TODO support stereo in
                track.trainer.step(X_left, Y)
                await websocket.send(
                    json.dumps({"train_state": track.trainer.params_and_loss()})
                )
            await asyncio.sleep(0.01)  # boo
        except websockets.ConnectionClosed:
            print("ws terminated")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAXdsp server")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO))

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    start_server = websockets.serve(
        register_websocket, "localhost", 8765, ssl=ssl_context
    )
    asyncio.get_event_loop().run_until_complete(start_server)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
