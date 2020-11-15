import jax.numpy as jnp

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
