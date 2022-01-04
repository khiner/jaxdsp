import argparse
import asyncio
import json
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

track_for_client_uid = {}
peer_connections = set()

int_max = np.iinfo(np.int16).max


class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.track = None
        self.params, self.state = None, None
        self.processor_names = None
        self.is_estimating = False
        self.trainer = IterativeTrainer()
        self.train_stack = deque([], MAX_TRAIN_FRAMES_PER_CLIENT)
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
        self.sample_rate = None  # cached and compared on each frame

    @trace
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
            self.update_sample_rate_state()
            # Don't pass any params to the trainer - that would be cheating ;)
            self.trainer.processor_names = processor_names
            self.trainer.set_carry((None, self.state))

    def set_loss_options(self, loss_options):
        self.trainer.set_loss_options(loss_options)

    def set_optimizer_options(self, optimizer_options):
        self.trainer.set_optimizer_options(optimizer_options)

    def start_estimating(self):
        self.is_estimating = True

    def stop_estimating(self):
        self.is_estimating = False

    def update_sample_rate_state(self):
        if self.sample_rate is not None:
            set_state_recursive(self.state, "sample_rate", self.sample_rate)

    # Takes a 2d array and returns a processed 2d array
    def process(self, X, sample_rate):
        X_left = X[0]  # TODO handle stereo in

        if self.params and self.state:
            if self.sample_rate != sample_rate:
                self.sample_rate = sample_rate
                self.update_sample_rate_state()
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
            if self.is_estimating:
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


def get_track_for_client_uid(client_uid):
    track = track_for_client_uid.get(client_uid)
    if not track:
        raise Exception(f"No track found for client UID {client_uid}")

    return track


def get_track_for_request(request):
    return get_track_for_client_uid(request.match_info.get("client_uid"))


async def index(request):
    return web.Response(
        content_type="text/plain",
        text="Use the `/offer` endpoint to negotiate a WebRTC peer connection.",
    )


async def offer(request):
    client_uid = str(uuid.uuid4())
    audio_transform_track = AudioTransformTrack()

    peer_connection = RTCPeerConnection()
    peer_connections.add(peer_connection)
    peer_connection_id = "PeerConnection(%s)" % uuid.uuid4()

    @peer_connection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = peer_connection.iceConnectionState
        print("ICE connection state is %s", state)
        if state == "failed" or state == "closed":
            await peer_connection.close()
            peer_connections.discard(peer_connection)

    @peer_connection.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        print("Track %s received", track.kind)

        audio_transform_track.track = track
        track_for_client_uid[client_uid] = audio_transform_track
        peer_connection.addTrack(audio_transform_track)

        @track.on("ended")
        async def on_ended():
            print("Track %s ended", track.kind)
            del track_for_client_uid[client_uid]

    # handle offer
    params = await request.json()
    offer_description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await peer_connection.setRemoteDescription(offer_description)

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


async def get_state(request):
    track = get_track_for_request(request)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "processor_definitions": [
                    serialize_processor(processor) for processor in ALL_PROCESSORS
                ],
                "optimizer_definitions": [
                    create_optimizer(definition.NAME).serialize()
                    for definition in all_optimizer_definitions
                ],
                "optimizer": track.trainer.optimizer.serialize(),
                "processors": [
                    serialize_processor(
                        processor_by_name[processor_name], processor_params
                    )
                    for processor_name, processor_params in zip(
                        track.trainer.processor_names,
                        track.trainer.params,
                    )
                ]
                if track.trainer.processor_names and track.trainer.params
                else None,
                "loss_options": track.trainer.loss_options.serialize(),
            }
        ),
    )


async def set_state(request):
    track = get_track_for_request(request)
    state = await request.json()
    if "processors" in state:
        track.set_graph_config(state["processors"])
    if "loss_options" in state:
        loss_options = state["loss_options"]
        track.set_loss_options(
            LossOptions(
                weights=loss_options.get("weights"),
                distance_types=loss_options.get("distance_types"),
                fft_sizes=loss_options.get("fft_sizes"),
            )
        )
    if "optimizer" in state:
        track.set_optimizer_options(state["optimizer"])

    return web.Response(status=201)


async def start_estimating(request):
    track = get_track_for_request(request)
    track.start_estimating()
    return web.Response(status=201)


async def stop_estimating(request):
    track = get_track_for_request(request)
    track.stop_estimating()
    return web.Response(status=201)


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

    track = get_track_for_client_uid(client_uid)
    train_stack = track.train_stack
    while True:
        try:
            heartbeat = {}
            if track.is_estimating and track.trainer:
                if len(train_stack) > 0:
                    train_pair = train_stack.pop()
                    X, Y = train_pair
                    X_left = X[0]  # TODO support stereo in
                    track.trainer.step(X_left, Y)
                heartbeat["train_events"] = track.trainer.get_events_serialized()
                track.trainer.clear_events()
            heartbeat["trace_events"] = tracer.get_events_serialized()
            tracer.clear_events()
            await websocket.send(json.dumps(heartbeat))
            await asyncio.sleep(0.05)  # boo
        except websockets.ConnectionClosed:
            print("ws terminated")
            break


def create_ssl_context(cert_file, key_file):
    if not cert_file:
        return None

    context = ssl.SSLContext()
    context.load_cert_chain(cert_file, key_file)
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAXdsp server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765, help="Port for WebSocket server (default: 8765)"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")

    args = parser.parse_args()

    ssl_context = create_ssl_context(args.cert_file, args.key_file)
    start_ws_server = websockets.serve(register_websocket, "0.0.0.0", port=args.ws_port, ssl=ssl_context)
    asyncio.get_event_loop().run_until_complete(start_ws_server)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    router = app.router
    router.add_routes(
        [
            web.get("/", index),
            web.post("/offer", offer),
            web.post("/start_estimating/{client_uid}", start_estimating),
            web.post("/stop_estimating/{client_uid}", stop_estimating),
            web.post("/state/{client_uid}", set_state),
            web.get("/state/{client_uid}", get_state),
        ]
    )

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(expose_headers="*", allow_headers="*")
        },
    )
    for route in list(router.routes()):
        cors.add(route)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
