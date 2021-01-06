import jax.numpy as jnp
from jax import jit, lax
from jax.ops import index_update

NAME = 'Allpass Filter'

def init_params(feedback=0.0):
    return {
        'feedback': feedback,
    }

def init_state(buffer_size=20):
    return {
        'buffer': jnp.zeros(buffer_size),
        'buffer_index': 0,
        'filter_store': 0.0,
    }

def default_target_params():
    return init_params(0.5)

@jit
def tick(carry, x):
    params = carry['params']
    state = carry['state']

    buffer_out = state['buffer'][state['buffer_index']]
    state['buffer'] = index_update(state['buffer'], state['buffer_index'], x + buffer_out * params['feedback'])
    state['buffer_index'] += 1
    state['buffer_index'] %= state['buffer'].size
    out = -x + buffer_out
    return carry, out

@jit
def tick_buffer(carry, X):
    return lax.scan(tick, carry, X)
