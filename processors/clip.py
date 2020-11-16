import jax.numpy as jnp

def init_params(shape):
    return jnp.asarray([[-1.0, 1.0]])

def init_state_from_params(params):
    return ()

def tick(x, params, state):
    min_max, = params
    return jnp.clip(x, min_max[0], min_max[1]), ()
