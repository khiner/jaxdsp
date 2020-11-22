import jax.numpy as jnp

NAME = 'clip'

def create_params_target():
    return {
        'min': -0.8,
        'max': 0.8,
    }

def init_params():
    return {
        'min': -1.0,
        'max': 1.0,
    }

def init_state_from_params(params):
    return ()

def tick(x, params, state):
    return jnp.clip(x, params['min'], params['max']), ()
