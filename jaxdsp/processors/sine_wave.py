import jax.numpy as jnp
from jax import jit

from jaxdsp.processors.base import default_param_values
from jaxdsp.param import Param

NAME = "Sine Wave"
PARAMS = [
    Param("frequency_hz", 400.0, 0.0, 16_000.0),
]
PRESETS = {}

SAMPLE_RATE = 16_000.0
BUFFER_SIZE = int(SAMPLE_RATE)
t = jnp.linspace(0, BUFFER_SIZE / SAMPLE_RATE, BUFFER_SIZE)

def init_state():
    return {"sample_rate": SAMPLE_RATE}


def init_params():
    return default_param_values(PARAMS)


def default_target_params():
    return {"frequency_hz": 443.0}


@jit
def tick(carry, x):
    params = carry["params"]
    return carry, jnp.sin(params["frequency_hz"] * 2 * jnp.pi * t)


@jit
def tick_buffer(carry, X):
    return tick(carry, X)
