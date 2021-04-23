import jax.numpy as jnp
from jax import jit, lax
from scipy import signal

from jaxdsp.param import Param
from jaxdsp.processors.base import Config, default_param_values

NAME = "IIR Filter"
# TODO how to handle array params in UI?
PARAMS = [
    Param("B", jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)])),
    Param("A", jnp.concatenate([jnp.array([1.0]), jnp.zeros(4)])),
]
PRESETS = {}


def config(length=5):
    B_target, A_target = signal.butter(length - 1, 0.5, "low")
    return Config(
        {
            "inputs": jnp.zeros(length),
            "outputs": jnp.zeros(length - 1),
        },
        default_param_values(PARAMS),
    )


@jit
def tick(carry, x):
    params = carry["params"]
    state = carry["state"]
    B = params["B"]
    A = params["A"]
    state["inputs"] = jnp.concatenate([jnp.array([x]), state["inputs"][0:-1]])
    y = B @ state["inputs"]
    if state["outputs"].size > 0:
        y -= A[1:] @ state["outputs"]
        # Don't optimize the output gain, since it's not commonly used and constraining it to 1 helps training
        # Note that this makes the implementation technically incorrect. We can always uncomment if we want it back.
        # y /= A[0]
        state["outputs"] = jnp.concatenate([jnp.array([y]), state["outputs"][0:-1]])
    return carry, y


@jit
def tick_buffer(carry, X):
    return lax.scan(tick, carry, X)
