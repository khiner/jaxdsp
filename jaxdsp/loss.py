import jax.numpy as jnp

def mse(X, Y):
    return ((Y - X) ** 2).mean()

def mae(X, Y):
    return jnp.abs(Y - X).mean()

# Based on https://github.com/numpy/numpy/issues/2310#issuecomment-314265666
def correlation(X, Y):
    c = jnp.correlate((X - jnp.mean(X)) / (jnp.std(X) * len(X)), (Y - jnp.mean(Y)) / (jnp.std(Y)))[0]
    return 1 - (1 + c) / 2

# TODO error-to-signal ratio (ESR)
#   MSE normalized by signal energy: "You need to be more accurate when your signal is very quiet"
