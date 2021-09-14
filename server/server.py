import argparse
import asyncio
import json
import logging
import ssl
import uuid
from collections import deque

import aiohttp_cors
import numpy as np
import websockets
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from jaxdsp import processor_graph
from jaxdsp.loss import LossOptions
from jaxdsp.optimizers import create_optimizer, all_optimizer_definitions
from jaxdsp.processors import (
    allpass_filter,
    clip,
    delay_line,
    biquad_lowpass,
    sine_wave,
    freeverb,
    serialize_processor,
    processor_by_name,
    graph_config_to_carry,
    processor_names_from_graph_config,
    set_state_recursive,
)
from jaxdsp.training import IterativeTrainer
from jaxdsp import tracer
from jaxdsp.tracer import trace

ALL_PROCESSORS = [allpass_filter, clip, delay_line, biquad_lowpass, sine_wave, freeverb]
# Training frame pairs are queued up for each client, limited to this cap:
MAX_TRAIN_FRAMES_PER_CLIENT = 100

logger = logging.getLogger("pc")
track_for_client_uid = {}
peer_connections = set()

int_max = np.iinfo(np.int16).max


class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track, train_stack):
        super().__init__()
        self.track = track
        self.params, self.state = None, None
        self.processor_names = None
        self.is_estimating_params = False
        self.trainer = IterativeTrainer()
        self.train_stack = train_stack
        self.previous_frame = None

        # Accumulate `packets_per_buffer` packets before processing them all at once,
        # both for the "forward pass" of processing incoming audio, and for the "backward pass"
        # of calculating gradients if the client for this track has estimation enabled.
        #
        # Downside: Introduces `packets_per_buffer - 1` packets of latency.
        #
        # Upside: Larger buffer sizes are faster to process per-sample because we
        # get more benefit from JAX's vectorization, and incur less round-trip cost
        # to the XLA code executed on the CPU/GPU (both for the forward pass and the gradient pass).
        # Estimation should generally be more stable with larger buffers.
        # Also, crucially for estimation, we can increase the maximum FFT size in the
        # multi-scale spectral loss.
        #
        # (Note: the opus 44100 Hz negotiation with both client and server on my machine only seems
        # to support 20ms packet sizes, otherwise this work would best be done in the RTC negotiation.)
        self.packets_per_buffer = 3  # TODO allow changing in realtime by client
        self.accumulated_packets = []
        self.processed_packets = []

    def set_graph_config(self, graph_config):
        if not graph_config:
            self.params, self.state = None, None
            self.trainer.set_graph_config(None)
            return

        processor_names = processor_names_from_graph_config(graph_config)
        # This is also the path to update processor params, regardless of whether the processor has changed.
        self.params, state = graph_config_to_carry(graph_config)
        if self.processor_names != processor_names:
            self.processor_names = processor_names
            self.state = state
            # Don't pass any params to the trainer - that would be cheating ;)
            self.trainer.processor_names = processor_names
            self.trainer.set_carry((None, self.state))

    def set_loss_options(self, loss_options):
        self.trainer.set_loss_options(loss_options)

    def set_optimizer_options(self, optimizer_options):
        self.trainer.set_optimizer_options(optimizer_options)

    def start_estimating_params(self):
        self.is_estimating_params = True

    def stop_estimating_params(self):
        self.is_estimating_params = False

    # Takes a 2d array and returns a processed 2d array
    def process(self, X, sample_rate):
        X_left = X[0]  # TODO handle stereo in

        if self.params and self.state:
            # TODO can this be done once on negotiate/processor change, instead of each frame?
            #  or, maybe it can be passed as another arg to tick_buffer
            set_state_recursive(self.state, "sample_rate", sample_rate)
            carry, Y = processor_graph.tick_buffer(
                (self.params, self.state), X_left, self.processor_names
            )
        else:
            carry, Y = (None, None), X_left

        self.state = carry[1]

        return np.array([Y, Y]) if Y.ndim == 1 else np.asarray(Y)

    @trace
    async def recv(self):
        frame = await self.track.recv()
        num_channels = len(frame.layout.channels)
        assert (
            frame.format.is_packed
        ), "Processing assumes frames are packed, but frame is planar"
        assert num_channels == 2, "Processing assumes frames have 2 channels"

        X_interleaved = (
            np.frombuffer(frame.planes[0], dtype=np.int16).astype(np.float32) / int_max
        )
        X_deinterleaved = [
            X_interleaved[channel_num::num_channels]
            for channel_num in range(num_channels)
        ]

        self.accumulated_packets.append(X_deinterleaved)
        if len(self.accumulated_packets) == self.packets_per_buffer:
            X = np.hstack(self.accumulated_packets)
            Y = self.process(X, frame.sample_rate)
            self.processed_packets = np.split(Y, self.packets_per_buffer, axis=1)
            self.accumulated_packets = []
            if self.is_estimating_params:
                self.train_stack.append([X, Y])

        if len(self.processed_packets) > 0:
            Y_deinterleaved = self.processed_packets.pop(0)
            Y_interleaved = np.empty(
                (Y_deinterleaved.shape[1] * 2,), dtype=Y_deinterleaved.dtype
            )
            Y_interleaved[0::2] = Y_deinterleaved[0]
            Y_interleaved[1::2] = Y_deinterleaved[1]

            out_samples = (Y_interleaved * int_max).astype(np.int16)
        else:
            # Fill with silence while waiting to receive enough packets to process.
            # Should only hit this case for the first `packets_per_buffer - 1` packets.
            out_samples = np.zeros(X_interleaved.size, dtype="int16")

        frame.planes[0].update(out_samples)
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
            if message == "get_state":
                channel.send(
                    json.dumps(
                        {
                            "processor_definitions": [
                                serialize_processor(processor)
                                for processor in ALL_PROCESSORS
                            ],
                            "optimizer_definitions": [
                                create_optimizer(definition.NAME).serialize()
                                for definition in all_optimizer_definitions
                            ],
                            "optimizer": audio_transform_track.trainer.optimizer.serialize(),
                            "processors": [
                                serialize_processor(
                                    processor_by_name[processor_name], processor_params
                                )
                                for processor_name, processor_params in zip(
                                    audio_transform_track.trainer.processor_names,
                                    audio_transform_track.trainer.params,
                                )
                            ]
                            if audio_transform_track.trainer.processor_names
                            and audio_transform_track.trainer.params
                            else None,
                            "loss_options": audio_transform_track.trainer.loss_options.serialize(),
                        }
                    )
                )
            elif message == "start_estimating_params":
                audio_transform_track.start_estimating_params()
            elif message == "stop_estimating_params":
                audio_transform_track.stop_estimating_params()
            else:
                message_dict = json.loads(message)
                if "processors" in message_dict:
                    audio_transform_track.set_graph_config(message_dict["processors"])
                if "loss_options" in message_dict:
                    loss_options = message_dict["loss_options"]
                    audio_transform_track.set_loss_options(
                        LossOptions(
                            weights=loss_options.get("weights"),
                            distance_types=loss_options.get("distance_types"),
                            fft_sizes=loss_options.get("fft_sizes"),
                        )
                    )
                optimizer_options = message_dict.get("optimizer")
                if optimizer_options:
                    audio_transform_track.set_optimizer_options(optimizer_options)

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
    while client_uid is None:
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
            heartbeat = {"tracer": tracer.get_json()}
            tracer.clear()
            if track.is_estimating_params and track.trainer:
                heartbeat["trainer"] = track.trainer.get_state()
            await websocket.send(json.dumps(heartbeat))
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
