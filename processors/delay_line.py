# The backprop is simply not working well for this.
# This is not surprising, as there are multiple parameters interplaying
# with potentially hundreds or thousands of timesteps before getting any feedback
# from a gradient nudge.
# Another potential approach to a differentiable delay is as a FIR filter
# with the first coefficient acting as the dry param, and the only other non-zero
# coefficient being the delay tap. This translates to three params:
#  1) a value for the first coefficient (the "dry amound")
#  2) a coefficient index for the "delay amount"
#  3) a coefficient value corresponding to this index (the "wet amount")
# This is just the existing fir_filter with different (more constrained) parameters.
# Of course this is incredibly inefficient from a DSP point of view, but maybe
# not much less performant for gradient descent.
# It's too bad, though. A big part of the value add for this project is supposed to be
# being able to implement DSP functions almost as you normally would. If I can't get the
# approach in this file working, hopefully it will stand as an exeption to this rule.

import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.ops import index, index_update

NAME = 'Delay Line'

MAX_DELAY_LENGTH_SAMPLES = 500 # 44_100

def init_params():
    return {
        'wet_amount': 0.0,
        'delay_length_normalized': 200.0 / MAX_DELAY_LENGTH_SAMPLES,
    }

def init_state():
    return {
        'delay_line': jnp.zeros(MAX_DELAY_LENGTH_SAMPLES),
        'read_sample': 0.0,
        'write_sample': 0,
    }


def create_params_target():
    return {
        'wet_amount': 1.0,
        'delay_length_normalized': 100.0 / MAX_DELAY_LENGTH_SAMPLES,
    }

@jit
def tick(carry, x):
    params = carry['params']
    state = carry['state']

    wet_amount = params['wet_amount']

    write_sample = state['write_sample']
    read_sample = state['read_sample']

    # delay_line[write_sample] = x
    state['delay_line'] = index_update(state['delay_line'], index[write_sample], x)
    delay_line = state['delay_line']

    read_sample_floor = read_sample.astype(int) # int(read_sample)
    interp = read_sample - read_sample_floor
    y = interp * delay_line[read_sample_floor]
    y += (1 - interp) * delay_line[(read_sample_floor + 1) % delay_line.size]

    state['write_sample'] += 1
    state['write_sample'] %= delay_line.size
    state['read_sample'] += 1
    state['read_sample'] %= delay_line.size

    out = x * (1 - wet_amount) + y * wet_amount
    return carry, out

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    state['read_sample'] = (state['write_sample'] - jnp.clip(params['delay_length_normalized'], 0, 1) * MAX_DELAY_LENGTH_SAMPLES) % state['delay_line'].size
    return lax.scan(tick, carry, X)[1]
