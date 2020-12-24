import jax.numpy as jnp
from jax import jit

NAME = 'Clip'

def init_state():
    return {}

def init_params():
    return {
        'min': -0.9,
        'max': 0.9,
    }

def create_params_target():
    return {
        'min': -0.5,
        'max': 0.5,
    }

@jit
def tick(carry, x):
    params = carry['params']
    return jnp.clip(x, params['min'], params['max'])

@jit
def tick_buffer(carry, X):
    return tick(carry, X)
