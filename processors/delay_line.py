# The backprop is simply not working well for this.
# This is not surprising, as there are multiple parameters interplaying
# with potentially hundreds or thousands of timesteps before getting any feedback
# from a gradient nudge.
#
# It does really well when the init and target delay length are within one sample
# of each other, and MAX_DELAY_LENGTH_SAMPLES is small. But the ability of the optimizer
# to find the correct gradient abruptly ends at the +/- 1 sample boundary.
# This is the case with/without param normalization to [0,1] for the delay_length param.
#
# Another potential approach to a differentiable delay is as an IIR filter
# with the first coefficient acting as the dry param, and the only other non-zero
# coefficient being the delay tap. This translates to three params:
#  1) a value for the first coefficient (the "dry amount")
#  2) a coefficient index for the "delay amount"
#  3) a coefficient value corresponding to this index (the "wet amount")
# This is just the existing iir_filter with different (more constrained) parameters.
# Of course, this is incredibly inefficient.
# It's too bad, though. A big part of the value add for this project is supposed to be
# being able to implement DSP functions almost as you normally would. If I can't get the
# approach in this file working, hopefully it will stand as an exeption to this rule.

import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.ops import index, index_update

NAME = 'Delay Line'

MAX_DELAY_LENGTH_SAMPLES = 100 # 44_100

def init_params():
    return {
        'wet_amount': 1.0,
        'delay_samples': 9.2,
    }

def init_state():
    return {
        'delay_line': jnp.zeros(MAX_DELAY_LENGTH_SAMPLES),
        'read_sample': 0.0,
        'write_sample': 0.0,
    }


def create_params_target():
    return {
        'wet_amount': 0.5,
        'delay_samples': 10.0,
    }

@jit
def tick(carry, x):
    params = carry['params']
    state = carry['state']

    write_sample = state['write_sample']
    read_sample = state['read_sample']

    state['delay_line'] = index_update(state['delay_line'], index[write_sample].astype('int32'), x)

    read_sample_floor = read_sample.astype('int32')
    interp = read_sample - read_sample_floor
    y = (1 - interp) * state['delay_line'][read_sample_floor]
    y += (interp) * state['delay_line'][(read_sample_floor + 1) % MAX_DELAY_LENGTH_SAMPLES]

    state['write_sample'] += 1
    state['write_sample'] %= MAX_DELAY_LENGTH_SAMPLES
    state['read_sample'] += 1
    state['read_sample'] %= MAX_DELAY_LENGTH_SAMPLES

    out = x * (1 - params['wet_amount']) + y * params['wet_amount']
    return carry, out

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    state['read_sample'] = (state['write_sample'] - jnp.clip(params['delay_samples'], 0, MAX_DELAY_LENGTH_SAMPLES)) % MAX_DELAY_LENGTH_SAMPLES
    return lax.scan(tick, carry, X)[1]
