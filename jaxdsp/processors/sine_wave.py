import jax.numpy as jnp
from jax import jit

from jaxdsp.processors.constants import DEFAULT_SAMPLE_RATE
from jaxdsp.param import Param

NAME = "Sine Wave"
PARAMS = [
    Param("frequency_hz", 440.0, 55.0, 7_040, log_scale=True),
]
PRESETS = {}


def init_state():
    return {"phase_radians": 0.0, "sample_rate": DEFAULT_SAMPLE_RATE}


@jit
def tick(carry, x):
    raise "single-sample tick method not implemented for sine_wave"


@jit
def tick_buffer(carry, X):
    params, state = carry
    # Add an extra sample to determine phase for start of next buffer.
    t = jnp.arange(X.size + 1) / state["sample_rate"]
    x = params["frequency_hz"] * 2 * jnp.pi * t + state["phase_radians"]
    state["phase_radians"] = x[-1] % (2 * jnp.pi)
    return carry, jnp.sin(x[:-1])
