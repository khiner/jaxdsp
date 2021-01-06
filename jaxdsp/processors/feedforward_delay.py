# `delay_samples` param is not reliably optimizable. See `delay_line.py` for more details.

import numpy as np
import jax.numpy as jnp
from jax import jit

NAME = 'Feedforward Delay'

MAX_DELAY_SIZE_SAMPLES = 44_100

def init_params(wet_amount=1.0, delay_samples=6.5):
    return {
        'wet_amount': wet_amount,
        'delay_samples': delay_samples,
    }

def init_state():
    return {}

def default_target_params():
    return init_params(0.5, 6.0)

def tick(carry, x):
    raise 'single-sample tick method not implemented for feedforward delay'

@jit
def tick_buffer(carry, X):
    params = carry['params']
    state = carry['state']
    delay_samples = params['delay_samples']
    remainder = delay_samples - jnp.floor(delay_samples)
    X_linear_interp = (1 - remainder) * X + remainder * jnp.concatenate([jnp.array([0]), X[:X.size - 1]])
    Y = jnp.where(jnp.arange(X.size) >= delay_samples, jnp.roll(X_linear_interp, delay_samples.astype('int32')), 0)
    return carry, X * (1 - params['wet_amount']) + Y * params['wet_amount']
