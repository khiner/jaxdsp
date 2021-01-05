# JAXdsp Server

Python WebRTC server for applying audio effects and learning effect parameters from audio examples.

Based on [the aiortc server example](https://github.com/aiortc/aiortc/tree/main/examples/server).

## Running

Install the required packages:

```console
$ pip install aiohttp aiortc opencv-python
```

Start the server:

```console
$ python server.py
```

This will create an HTTP server which you can connect to from your browser at:

http://localhost:8080

Once you click `Start` the browser will send the audio and video from its
webcam to the server.

The server will play a pre-recorded audio clip and send the received video back
to the browser, optionally applying a transform to it.

In parallel to media streams, the browser sends a 'ping' message over the data
channel, and the server replies with 'pong'.
