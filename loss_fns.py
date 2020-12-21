from jax import jit

@jit
def mse(X, Y):
    return ((Y - X) ** 2).mean()

@jit
def mae(X, Y):
    return jnp.abs(Y - X).mean()
