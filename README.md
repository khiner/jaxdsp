# JAXdsp

**Still working on this and will make a blog post soon!**

Fast, differentiable audio processors on the CPU or GPU, controlled from the browser.

The goal is to parameterize audio graphs in real-time to resemble incoming audio streams.

Built with [JAX](https://github.com/google/jax), WebRTC, WebSockets and React.

```shell
pip install git+https://github.com/cifkao/jax-spectral.git@main#egg=jax-spectral
```

## Server

```shell
$ cd server
$ python server.py
```

## Client

```shell
$ cd client
$ npm ci
$ npm run build:dev
```

## App

```shell
$ cd app
$ npm ci
$ npm start
```
