# Fractional delay line using linear interpolation

# The `delay_samples` param cannot be optimized outside of the +/- 1 sample range between
# init and target.
# This is because the param maps directly to an `index_update` operation,
# which requires converting the parameter to an integer, which is a non-differentiable operation.
# Linear interpolation helps by allowing successful optimization within the +/- 1 sample
# range, but I have not found a solution that allows the gradient signal to propagate
# across the entire indexing range.
# (See "Differentiable array indexing" notebook for more details.)


import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.ops import index, index_update

from jaxdsp.param import Param

MAX_DELAY_LENGTH_SAMPLES = 44100

NAME = "Delay Line"
PARAMS = [
    Param("wet", 1.0),
    Param("delay_samples", 0.9, 0.0, float(MAX_DELAY_LENGTH_SAMPLES)),
]
PRESETS = {}


def init_state():
    return {
        "delay_line": jnp.zeros(MAX_DELAY_LENGTH_SAMPLES),
        "read_sample": 0.0,
        "write_sample": 0.0,
    }


@jit
def tick(carry, x):
    params, state = carry

    write_sample = state["write_sample"]
    read_sample = state["read_sample"]

    state["delay_line"] = index_update(
        state["delay_line"], index[write_sample].astype("int32"), x
    )

    read_sample_floor = read_sample.astype("int32")
    interp = read_sample - read_sample_floor
    y = (1 - interp) * state["delay_line"][read_sample_floor]
    y += (interp) * state["delay_line"][
        (read_sample_floor + 1) % MAX_DELAY_LENGTH_SAMPLES
    ]

    state["write_sample"] += 1
    state["write_sample"] %= MAX_DELAY_LENGTH_SAMPLES
    state["read_sample"] += 1
    state["read_sample"] %= MAX_DELAY_LENGTH_SAMPLES

    out = x * (1 - params["wet"]) + y * params["wet"]
    return carry, out


@jit
def tick_buffer(carry, X):
    params, state = carry
    state["read_sample"] = (
        state["write_sample"]
        - jnp.clip(params["delay_samples"], 0, MAX_DELAY_LENGTH_SAMPLES)
    ) % MAX_DELAY_LENGTH_SAMPLES
    return lax.scan(tick, carry, X)
