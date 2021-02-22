import jax.numpy as jnp
from jax import jit

from jaxdsp.processors.base import Config, default_param_values
from jaxdsp.param import Param

NAME = "Clip"
PARAMS = [Param("min", -1.0, -1.0, 1.0), Param("max", 1.0, -1.0, 1.0)]
PRESETS = {}


def config():
    return Config({}, default_param_values(PARAMS), {"min": -0.5, "max": 0.5})


@jit
def tick(carry, x):
    params = carry["params"]
    return carry, jnp.clip(x, params["min"], params["max"])


@jit
def tick_buffer(carry, X):
    return tick(carry, X)
