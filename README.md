# JAXdsp

**I'm still working on improvements and an interactive browser client, after which I'll write up a blog post. In the
meantime, the `docs` directory has several Jupyter notebooks with plenty of examples.
The `Differentiable Audio Processors` shows many real DSP examples, along with loss/parameter time-series.** _(`Differentiable Array Indexing` and `Log Normal Distribution` are just working notes.)_

Fast, differentiable audio processors on the CPU or GPU, controlled from the browser.

The goal is to parameterize audio graphs, in real-time, to produce an audio stream resembling incoming audio.

Built with [JAX](https://github.com/google/jax), WebRTC, WebSockets and React.

_**Note:** The `requirements.txt` file at the root of this repo was generated with `pip freeze > requirements.txt`, and
it likely contains more than what's strictly needed!_

## Server

```shell
$ cd server
$ python server.py
```

## Client

_**Note:** For local development of the client within the test `/app` (see below), the client is included as a
local `file:../client` dependency, and included in the babel transpilation for the `/app` build (
see `app/craco.config.js`)._

_`build:dev` in the client just symlinks the `dist/jaxdsp-client.js` target to the root `index.js` file._

```shell
$ cd client
$ npm ci
$ npm run build # or, for a development build: `build:dev`
```

## App

```shell
$ cd app
$ npm ci
$ npm start
```
