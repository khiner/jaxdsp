import jax.numpy as jnp
from jax import jit, lax

from jaxdsp.param import Param

NAME = "FIR Filter"
# TODO how to handle array params in UI?
PARAMS = [Param("B", jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)]))]
PRESETS = {}


def init_state(length=4):
    return {"inputs": jnp.zeros(length)}


@jit
def tick(carry, x):
    params, state = carry
    B = params["B"]
    state["inputs"] = jnp.concatenate([jnp.array([x]), state["inputs"][0:-1]])
    y = B @ state["inputs"]
    return carry, y


@jit
def tick_buffer(carry, X):
    params, _ = carry
    B = params["B"]
    return carry, jnp.convolve(X, B)[: -(B.size - 1)]
    # Impossibly, the following seems to perform about the exact same or even faster for large N?! O_O
    # return lax.scan(tick, carry, X)[1]
