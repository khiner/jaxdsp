import jax.numpy as jnp
from jax import jit

from jaxdsp.processors.base import default_param_values
from jaxdsp.param import Param

NAME = "Clip"
PARAMS = [Param("min", -1.0, -1.0, 1.0), Param("max", 1.0, -1.0, 1.0)]
PRESETS = {}


def init_state():
    return {}


def init_params():
    return default_param_values(PARAMS)


def default_target_params():
    return {"min": -0.5, "max": 0.5}


@jit
def tick(carry, x):
    params = carry["params"]
    return carry, jnp.clip(x, params["min"], params["max"])


@jit
def tick_buffer(carry, X):
    return tick(carry, X)
