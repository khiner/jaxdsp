import jax.numpy as jnp
from jax import jit

NAME = 'Clip'

def init_state():
    return {}

def init_params(min_value=-0.9, max_value=0.9):
    return {
        'min': min_value,
        'max': max_value,
    }

def default_target_params():
    return init_params(-0.5, 0.5)

@jit
def tick(carry, x):
    params = carry['params']
    return carry, jnp.clip(x, params['min'], params['max'])

@jit
def tick_buffer(carry, X):
    return tick(carry, X)
