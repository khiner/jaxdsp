# Goal: Update jnp array at an index calculated from a parameter being differentiated.


# [Error taking gradient through ops.index_update](https://github.com/google/jax/issues/1104)
#   https://github.com/google/jax/pull/1056

import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from jax.ops import index, index_update, index_add

import sys
# sys.path.append('./')
# import iir_filter

NAME = 'Index Update'

MAX_DELAY_LENGTH = 8

def init_state():
    return {}

def init_params():
    return {
        'sample_index' : 4.0,
    }

def create_params_target():
    return {
        'sample_index': 3.0,
    }


# TODO calculate sigma based on MAX_DELAY_LENGTH for good tradeoff between accuracy and convergence
@jit
def gaussian(x, x0, sigma):
    return jnp.exp(-((x - x0) / sigma)**2 / 2)

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    X = index_update(state['A'], index[1:], gaussian(jnp.arange(MAX_DELAY_LENGTH - 1), params['sample_index'], 2.0))
#    X = index_update(X, index[1:], gaussian(jnp.arange(MAX_DELAY_LENGTH - 1), params['sample_index'], 2.0))
    return X
