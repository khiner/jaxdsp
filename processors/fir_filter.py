import jax.numpy as jnp

def create_params_target():
    return {
        'B': jnp.array([0.1, 0.7, 0.5, 0.6])
    }

def init_params():
    return {
        'B' : jnp.concatenate([jnp.array([1.0]), jnp.zeros(3)])
    }

def init_state_from_params(params):
    B = params['B']
    inputs = jnp.zeros(B.size)
    return (inputs,)

def tick(x, params, state):
    B = params['B']
    inputs, = state
    inputs = jnp.concatenate([jnp.array([x]), inputs[0:-1]])
    y = B @ inputs
    return y, (inputs,)
