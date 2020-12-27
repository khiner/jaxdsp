# Suffers from the same optimization issue as delay_line.
# See delay_line.py for more details.
import jax.numpy as jnp
from jax import jit, lax
from jax.ops import index, index_update

import sys
sys.path.append('./')
from iir_filter import tick

NAME = 'IIR Delay'

MAX_DELAY_LENGTH = 100

def init_state():
    return {
        'inputs': jnp.zeros(1),
        'outputs': jnp.zeros(MAX_DELAY_LENGTH - 1),
    }

def init_params():
    return {
        'delay_length_samples' : 20.1,
    }

def default_target_params():
    return {
        'delay_length_samples': 21.0,
    }

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    delay_length_samples = jnp.clip(params['delay_length_samples'], 0, MAX_DELAY_LENGTH - 1)
    delay_coefficient = -1.0

    i0 = jnp.floor(delay_length_samples)
    i1 = i0 + 1
    A = jnp.concatenate([jnp.array([1.0]), jnp.zeros(MAX_DELAY_LENGTH - 1)])
    A = index_update(A, i0.astype('int32'), delay_coefficient * (i1 - delay_length_samples))
    A = index_update(A, i1.astype('int32'), delay_coefficient * (delay_length_samples - i0))
    # Throw A and B coefficients in params to reuse the iir_filter::tick fn,
    # even though they're not being optimized.
    params['A'] = A
    params['B'] = jnp.array([1.0])
    return lax.scan(tick, carry, X)[1]

