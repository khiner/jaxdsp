import jax.numpy as jnp

from jaxdsp.processors import processor_by_name

# This does not actually support full graph connectivity.
# For simplicity, only series & parallel processing is supported.
# A wide variety of common DSP techniques can be implemented by nesting series and parallel processor groups.
# Keeping this restriction (at least for now) means the graph structure can be represented and serialized
# simply as a list, where each element can be a processor or a list.
# This also simplifies the UI since the "graph" connectivity is implicit in the positional order.
# Thus, we don't need to support arbitrary connectivity, and can instead use a simple drag-and-drop UX paradigm.

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
