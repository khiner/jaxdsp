import jax.numpy as jnp
from jax import jit
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
