import jax.numpy as jnp

class Clip():
    NAME = 'clip'

    def init_params():
        return {
            'min': -1.0,
            'max': 1.0,
        }

    def create_params_target():
        return {
            'min': -0.8,
            'max': 0.8,
        }

    def tick_buffer(self, X, params):
        return jnp.clip(X, params['min'], params['max'])

    def tick(self, x, params):
        return jnp.clip(x, params['min'], params['max'])
