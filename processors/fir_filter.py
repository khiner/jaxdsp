import jax.numpy as jnp

class FirFilter():
    NAME = 'fir_filter'

    def __init__(self):
        self.init_params = {
            'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(3)])
        }
        self.inputs = jnp.zeros(self.init_params['B'].size)

    def create_params_target(self):
        return {
            'B': jnp.array([0.1, 0.7, 0.5, 0.6])
        }

    def tick(self, x, params):
        B = params['B']
        self.inputs = jnp.concatenate([jnp.array([x]), self.inputs[0:-1]])
        y = B @ self.inputs
        return y
