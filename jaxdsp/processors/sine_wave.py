import jax.numpy as jnp
from jax import jit

from jaxdsp.processors.base import default_param_values
from jaxdsp.param import Param

NAME = "Sine Wave"
PARAMS = [
    Param("frequency_hz", 880.0, 0.0, 16_000.0),
    Param("phase", 0.0, 0.0, 2 * jnp.pi),
]
PRESETS = {}

SAMPLE_RATE = 44_100.0
BUFFER_SIZE = int(SAMPLE_RATE)
t = jnp.linspace(0, BUFFER_SIZE / SAMPLE_RATE, BUFFER_SIZE)

# http://devmaster.net/forums/topic/4648-fast-and-accurate-sinecosine/
@jit
def sine_approx(x):
    y = 4 / jnp.pi * x - 4 / (jnp.pi ** 2) * x * jnp.abs(x)
    return y * (0.225 * (jnp.abs(y) - 1) + 1)


def init_state():
    return {"sample_rate": SAMPLE_RATE}


def init_params():
    return default_param_values(PARAMS)


def default_target_params():
    return {"frequency_hz": 440.0, "phase": jnp.pi}


@jit
def tick(carry, x):
    params = carry["params"]
    x = (
        params["frequency_hz"] * 2 * jnp.pi + params["phase"]
    ) * t  # % 2 * jnp.pi - jnp.pi
    return carry, jnp.sin(x)  # sine_approx(x)


@jit
def tick_buffer(carry, X):
    return tick(carry, X)
