from jax import lax
import jax.numpy as jnp

def process(params, processor_class, X):
    processor = processor_class()
    # lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)
    return jnp.array([processor.tick(x, params) for x in X])

def process_buffer(params, processor_class, X):
    processor = processor_class()
    return processor.tick_buffer(X, params)

def process_serial(params, processor_class, processors, X):
    processor = processor_class(processors)
    return jnp.array([processor.tick(x, params) for x in X])
