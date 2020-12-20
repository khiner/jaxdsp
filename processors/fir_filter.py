import jax.numpy as jnp

class FirFilter():
    NAME = 'fir_filter'

    def init_params(length=4):
        return {
            'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)])
        }

    def create_params_target(length=4):
        return {
            'B': jnp.array([0.1, 0.7, 0.5, 0.6])
        }


    def __init__(self, length=4):
        self.inputs = jnp.zeros(length)


    def tick_buffer(self, X, params):
        B = params['B']
        return jnp.convolve(X, B)[:-(B.size - 1)]

    def tick(self, x, params):
        B = params['B']
        self.inputs = jnp.concatenate([jnp.array([x]), self.inputs[0:-1]])
        y = B @ self.inputs
        return y
