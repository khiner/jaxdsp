import jax.numpy as jnp

def init_state_from_params(params):
    B, A = params
    inputs = jnp.zeros(B.size)
    outputs = jnp.zeros(A.size - 1)
    return (inputs, outputs)

def tick(x, params, state):
    inputs, outputs = state
    B, A = params
    inputs = jnp.concatenate([jnp.array([x]), inputs[0:-1]])
    y = B @ inputs
    if outputs.size > 0:
        y -= A[1:] @ outputs
        y /= A[0]
        outputs = jnp.concatenate([jnp.array([y]), outputs[0:-1]])
    return y, (inputs, outputs)
