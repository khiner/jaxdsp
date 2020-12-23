import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from jax.ops import index, index_update

import sys
# sys.path.append('./')
# import iir_filter

NAME = 'IIR Delay'

MAX_DELAY_LENGTH = 100

def init_state():
    return {
        'inputs': jnp.zeros(1),
        'outputs': jnp.zeros(MAX_DELAY_LENGTH - 1),
        'B': jnp.array([1.0]),
        'A': jnp.concatenate([jnp.array([1.0]), jnp.zeros(MAX_DELAY_LENGTH - 1)]),
    }

def init_params():
    return {
        'delay_length_samples' : 20.0,
    }

def create_params_target():
    return {
        'delay_length_samples': 30.0,
    }

@jit
def tick(carry, x):
    state = carry['state']
    B = state['B']
    A = state['A']
    inputs = jnp.concatenate([jnp.array([x]), state['inputs'][0:-1]])
    outputs = state['outputs']
    y = B @ inputs
    if outputs.size > 0:
        y -= A[1:] @ outputs
        # Don't optimize the output gain, since it's not commonly used and constraining it to 1 helps training
        # Note that this makes the implementation technically incorrect. We can always uncomment if we want it back.
        # y /= A[0]
        outputs = jnp.concatenate([jnp.array([y]), outputs[0:-1]])
    state['inputs'] = inputs
    state['outputs'] = outputs
    return carry, y

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    delay_length_samples = jnp.clip(params['delay_length_samples'], 0, MAX_DELAY_LENGTH - 1)
    delay_coefficient = -1.0

    # Linear interpolation doesn't get a gradient signal across the entire indexing range.
    # (See "Differentiable array indexing" notebook for more details.)
    i0 = jnp.floor(delay_length_samples)
    i1 = i0 + 1
    state['A'] = index_update(state['A'], i0.astype('int32'), delay_coefficient * (i1 - delay_length_samples))
    state['A'] = index_update(state['A'], i1.astype('int32'), delay_coefficient * (delay_length_samples - i0))
    return lax.scan(tick, carry, X)[1]

