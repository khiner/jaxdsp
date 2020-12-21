import jax.numpy as jnp
from jax import jit, lax
from scipy import signal

NAME = 'iir_filter'

def init_state(length=5):
    return {
        'inputs': jnp.zeros(length),
        'outputs': jnp.zeros(length - 1),
    }

def init_params(length=5):
    return {
        'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)]),
        'A' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(length - 1)]),
    }

def create_params_target(length=5):
    B, A = signal.butter(length - 1, 0.5, 'low')
    return {
        'B': B,
        'A': A,
    }

@jit
def tick(carry, x):
    params = carry['params']
    state = carry['state']
    B = params['B']
    A = params['A']
    inputs = jnp.concatenate([jnp.array([x]), state['inputs'][0:-1]])
    outputs = state['outputs']
    y = B @ inputs
    if outputs.size > 0:
        y -= A[1:] @ state['outputs']
        y /= A[0]
        outputs = jnp.concatenate([jnp.array([y]), outputs[0:-1]])
    state['inputs'] = inputs
    state['outputs'] = outputs
    return carry, y

@jit
def tick_buffer(carry, X):
    return lax.scan(tick, carry, X)[1]

