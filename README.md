# JAXdsp Server

Python WebRTC server for applying audio effects and learning effect parameters from audio examples.

Note: lots of code taken from [the aiortc server example](https://github.com/aiortc/aiortc/tree/main/examples/server).

## Running

Install the required packages:

```console
$ pip install aiohttp aiohttp_cors aiortc opencv-python
```

Start the server:

```console
$ python server.py
```

This will create an HTTP server which you can connect to from your browser at [location](http://localhost:8080).
