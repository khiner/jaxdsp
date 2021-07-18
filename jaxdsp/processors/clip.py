import jax.numpy as jnp
from jax import jit

from jaxdsp.param import Param

NAME = "Clip"
PARAMS = [Param("min", -1.0, -1.0, 1.0), Param("max", 1.0, -1.0, 1.0)]
PRESETS = {}


def init_state():
    return {}


@jit
def tick(carry, x):
    params, _ = carry
    return carry, jnp.clip(x, params["min"], params["max"])


@jit
def tick_buffer(carry, X):
    return tick(carry, X)
