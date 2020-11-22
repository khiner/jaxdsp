import jax.numpy as jnp
from scipy import signal

class IirFilter():
    NAME = 'iir_filter'

    def __init__(self):
        self.init_params = {
            'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)]),
            'A' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)]),
        }
        self.inputs = jnp.zeros(self.init_params['B'].size)
        self.outputs = jnp.zeros(self.init_params['A'].size - 1)

    def create_params_target(self):
        B, A = signal.butter(4, 0.5, 'low')
        return {
            'B': B,
            'A': A,
        }

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
