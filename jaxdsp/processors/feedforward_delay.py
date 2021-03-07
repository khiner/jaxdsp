# `delay_samples` param is not reliably optimizable. See `delay_line.py` for more details.

import numpy as np
import jax.numpy as jnp
from jax import jit

from jaxdsp.param import Param
from jaxdsp.processors.base import Config, default_param_values

MAX_DELAY_SIZE_SAMPLES = 44100

NAME = "Feedforward Delay"
PARAMS = [
    Param("wet", 1.0),
    Param("delay_samples", 6.5, 0.0, float(MAX_DELAY_SIZE_SAMPLES)),
]
PRESETS = {}


def config():
    return Config(
        {},
        default_param_values(PARAMS),
        {"wet": 0.5, "delay_samples": 6.0},
    )


def tick(carry, x):
    raise "single-sample tick method not implemented for feedforward_delay"


@jit
def tick_buffer(carry, X):
    params = carry["params"]
    delay_samples = params["delay_samples"]
    remainder = delay_samples - jnp.floor(delay_samples)
    X_linear_interp = (1 - remainder) * X + remainder * jnp.concatenate(
        [jnp.array([0]), X[: X.size - 1]]
    )
    Y = jnp.where(
        jnp.arange(X.size) >= delay_samples,
        jnp.roll(X_linear_interp, delay_samples.astype("int32")),
        0,
    )
    return carry, X * (1 - params["wet"]) + Y * params["wet"]
