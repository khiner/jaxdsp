import jax.numpy as jnp

class Clip():
    NAME = 'clip'

    def __init__(self):
        self.init_params = {
            'min': -1.0,
            'max': 1.0,
        }

    def create_params_target(self):
        return {
            'min': -0.8,
            'max': 0.8,
        }

    def tick(self, x, params):
        return jnp.clip(x, params['min'], params['max'])
