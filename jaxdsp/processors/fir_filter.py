import jax.numpy as jnp
from jax import jit, lax

NAME = 'FIR Filter'

def init_state(length=4):
    return {
        'inputs': jnp.zeros(length)
    }

def init_params(length=4):
    return {
        'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)])
    }

def default_target_params(length=4):
    return {
        'B': jnp.array([0.1, 0.7, 0.5, 0.6])
    }

@jit
def tick(carry, x):
    state = carry['state']
    params = carry['params']
    B = params['B']
    state['inputs'] = jnp.concatenate([jnp.array([x]), state['inputs'][0:-1]])
    y = B @ state['inputs']
    return carry, y

@jit
def tick_buffer(carry, X):
    params = carry['params']
    B = params['B']
    return carry, jnp.convolve(X, B)[:-(B.size - 1)]
    # Impossibly, the following seems to perform about the exact same or even faster for large N?! O_O
    # return lax.scan(tick, carry, X)[1]
