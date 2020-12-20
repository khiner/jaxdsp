import jax.numpy as jnp
from jax import jit, lax
from scipy import signal

class IirFilter():
    NAME = 'iir_filter'

    def init_params(length=5):
        return {
            'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)]),
            'A' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)]),
        }

    def create_params_target(length=5):
        B, A = signal.butter(length - 1, 0.5, 'low')
        return {
            'B': B,
            'A': A,
        }


    def __init__(self, length=5):
        self.inputs = jnp.zeros(length)
        self.outputs = jnp.zeros(length - 1)

    def tick_buffer(self, X, params):
        return IirFilter.tick_buffer_scan(X, params, self.inputs, self.outputs)[1]

    def tick(self, x, params):
        B = params['B']
        A = params['A']
        self.inputs = jnp.concatenate([jnp.array([x]), self.inputs[0:-1]])
        y = B @ self.inputs
        if self.outputs.size > 0:
            y -= A[1:] @ self.outputs
            y /= A[0]
            self.outputs = jnp.concatenate([jnp.array([y]), self.outputs[0:-1]])
        return y

    @jit
    def tick_buffer_scan(X, params, inputs, outputs):
        init = {'B': params['B'], 'A': params['A'], 'inputs': inputs, 'outputs': outputs}
        return lax.scan(IirFilter.tick_scan, init, X)

    @jit
    def tick_scan(carry, x):
        B = carry['B']
        A = carry['A']
        inputs = jnp.concatenate([jnp.array([x]), carry['inputs'][0:-1]])
        outputs = carry['outputs']
        y = B @ inputs
        if outputs.size > 0:
            y -= A[1:] @ carry['outputs']
            y /= A[0]
            outputs = jnp.concatenate([jnp.array([y]), outputs[0:-1]])
        carry['inputs'] = inputs
        carry['outputs'] = outputs
        return carry, y