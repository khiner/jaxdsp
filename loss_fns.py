from jax import jit

def mse(X, Y):
    return ((Y - X) ** 2).mean()

def mae(X, Y):
    return jnp.abs(Y - X).mean()
