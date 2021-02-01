# JAXdsp

Fast, differentiable audio processors on the CPU or GPU.
Built with [JAX](https://github.com/google/jax).

## Server

```shell
$ cd server
$ python server.py
```

## Client

```shell
$ cd client
$ npm ci
$ npm start
```

## TODO

- In `jaxdsp_server`, provide control control over all training parameters (batch-size, step-size, optimizer, loss function, etc.) via data-channel messages from the client
- Send forward-pass/train step timing data-channel messages to client from `jaxdsp-server`
- Add `parallel_processors` processor to `jaxdsp`. Support nesting of `parallel_processors` and `serial_processors` as a structurally constrained audio graph. Thus, `jaxdsp` allows optimization over a single `processor`, but that processor can be an arbitrary nesting of serial/parallel/plain-old processors
  - Also update the `train` method that plots param/loss histories to use a more general nested grid pattern in `matplotlib` to match the graph structure (nested rows/columns)
- Use [react-flow](https://reactflow.dev/) library to implement `jaxdsp_client.monitor` and `jaxdsp_client.graph_editor`
  - `monitor` shows a real-time flow-diagram visualization of the full client/server process, including peerConnection details, data-channel messages, and live charts tracking all estimated parameter values & loss histories (using [nivo](https://nivo.rocks/line/) charting library)
  - `graph_editor` provides an editable graph, with parameter controls inside each node, with the ability to connect nodes in parallel and in serial (vertical & horizontal processor groups, respectively)
- Create docker env for `jaxdsp_server`
- Deploy `jaxdsp_server` docker to a2hosting instance
- Expand `jaxdsp` jupyter-notebook in `/docs` into more comprehensive tutorials
- GPU?!
  - run home server
  - add switch to `jaxdsp_client.monitor` to switch between cpu/gpu
- Add a perceptual loss function instead of mse (steal DDSP's multi-scale spectral loss fn)
- Write blog post
  - parameter initialization: starting from no effect initially
    - small changes to params has big effect in processors with feedback
    - also good for live performance settings
- Stretch: Replicate the Boss DS-1 pedal experiment from [Differentiable IIR Filters for Machine Learning Applications (2020)](https://www.dafx.de/paper-archive/details.php?id=rA_6fTdLky8YDvH03jdufw)
- Stretch: Work more on making delay-length in delay line differentiable
  - started in "Differentiable array indexing" notebook
  - show plots of optimization for different starting guesses, to help get a sense of why the gradient is blocked outside the +/- 1 range from initial delay-length param to target.
