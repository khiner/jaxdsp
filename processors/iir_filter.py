import jax.numpy as jnp
from scipy import signal

NAME = 'iir_filter'

def create_params_target():
    B, A = signal.butter(4, 0.5, 'low')
    return {
        'B': B,
        'A': A,
    }

def init_params():
    return {
        'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)]),
        'A' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)]),
    }

def init_state_from_params(params):
    B = params['B']
    A = params['A']
    inputs = jnp.zeros(B.size)
    outputs = jnp.zeros(A.size - 1)
    return (inputs, outputs)

def tick(x, params, state):
    B = params['B']
    A = params['A']
    inputs, outputs = state
    inputs = jnp.concatenate([jnp.array([x]), inputs[0:-1]])
    y = B @ inputs
    if outputs.size > 0:
        y -= A[1:] @ outputs
        y /= A[0]
        outputs = jnp.concatenate([jnp.array([y]), outputs[0:-1]])
    return y, (inputs, outputs)
