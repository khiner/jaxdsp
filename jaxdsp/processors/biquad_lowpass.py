# http://www.earlevel.com/main/2012/11/26/biquad-c-source-code/

import jax.numpy as jnp
from jax import jit, lax

from jaxdsp.param import Param

NAME = "BiQuad Lowpass Filter"
PARAMS = [Param("resonance", 0.7), Param("cutoff", 0.49, 0.0, 0.49)]
PRESETS = {}


def init_state():
    return {
        "a0": 1.0,
        "a1": 0.0,
        "a2": 0.0,
        "b1": 0.0,
        "b2": 0.0,
        "z1": 0.0,
        "z2": 0.0,
    }


@jit
def tick(carry, x):
    _, state = carry
    out = x * state["a0"] + state["z1"]
    state["z1"] = x * state["a1"] + state["z2"] - state["b1"] * out
    state["z2"] = x * state["a2"] - state["b2"] * out

    return carry, out


@jit
def tick_buffer(carry, X):
    params, state = carry
    K = jnp.tan(jnp.pi * params["cutoff"])
    norm = 1.0 / (1.0 + K / params["resonance"] + K * K)
    state["a0"] = K * K * norm
    state["a1"] = 2.0 * state["a0"]
    state["a2"] = state["a0"]
    state["b1"] = 2.0 * (K * K - 1) * norm
    state["b2"] = (1.0 - K / params["resonance"] + K * K) * norm
    return lax.scan(tick, carry, X)
