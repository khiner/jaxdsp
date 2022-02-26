# JAXdsp Server

Python WebRTC server for applying audio effects and learning effect parameters from audio examples.

_To run via Docker, see [the main JAXdsp readme](../README.md)._

## Install requirements

### MacOS

Install [aiortc requirements](https://github.com/aiortc/aiortc#os-x) & install packages:

```console
$ brew install ffmpeg@4 opus libvpx pkg-config
$ pip install -r requirements.txt
```

## Running

```console
$ python server.py
```

The server will be running at [location](http://localhost:8080).
