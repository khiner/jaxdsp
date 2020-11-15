import jax.numpy as jnp

def init_params(shape):
    size = shape[1]
    init = jnp.zeros(shape)
    if size >= 1:
        init = init.at[:,0].set(1.0)
    return init

def init_state_from_params(params):
    B, = params
    inputs = jnp.zeros(B.size)
    return (inputs,)

def tick(x, params, state):
    inputs, = state
    B, = params
    inputs = jnp.concatenate([jnp.array([x]), inputs[0:-1]])
    y = B @ inputs
    return y, (inputs,)
