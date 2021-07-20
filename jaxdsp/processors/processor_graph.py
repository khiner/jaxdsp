import jax.numpy as jnp

from jaxdsp.processors import processor_by_name


def tick_buffer_series(carry, X, processor_names):
    params, state = carry

    Y = X
    for i, processor_name in enumerate(processor_names):
        processor_state = state[i]
        processor_params = params[i]
        processor = processor_by_name[processor_name]
        processor_carry, Y = processor.tick_buffer(
            (processor_params, processor_state), Y
        )
        carry[1][i] = processor_carry[1]

    return carry, Y


def tick_buffer_parallel(carry, X, processor_names):
    params, state = carry

    Y = jnp.zeros(X.shape)
    for i, processor_name in enumerate(processor_names):
        processor_state = state[i]
        processor_params = params[i]
        processor = processor_by_name[processor_name]
        processor_carry, Y_i = processor.tick_buffer(
            (processor_params, processor_state), X
        )
        Y += Y_i
        carry[1][i] = processor_carry[1]

    return carry, Y
