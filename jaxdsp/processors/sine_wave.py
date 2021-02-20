import jax.numpy as jnp
from jax import jit

from jaxdsp.config import buffer_size, sample_rate
from jaxdsp.processors.base import default_param_values
from jaxdsp.param import Param

NAME = "Sine Wave"
PARAMS = [
    Param("frequency_hz", 400.0, 0.0, 16_000.0),
]
PRESETS = {}

t = jnp.linspace(0, buffer_size / sample_rate, buffer_size)


def init_state():
    return {}


def init_params():
    return default_param_values(PARAMS)


def default_target_params():
    return {"frequency_hz": 443.0}


@jit
def tick(carry, x):
    raise "single-sample tick method not implemented for sine_wave"


@jit
def tick_buffer(carry, X):
    params = carry["params"]
    return carry, jnp.sin(params["frequency_hz"] * 2 * jnp.pi * t)
