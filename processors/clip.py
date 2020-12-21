import jax.numpy as jnp
from jax import jit

NAME = 'clip'

def init_state():
    return {}

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

@jit
def tick(carry, x):
    params = carry['params']
    return jnp.clip(x, params['min'], params['max'])

@jit
def tick_buffer(carry, X):
    return tick(carry, X)
