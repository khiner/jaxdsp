import jax.numpy as jnp

from jaxdsp.processors import processor_by_name


# This module does not actually support full graph connectivity.
# For simplicity, only series & parallel processing is supported.
# A wide variety of common DSP techniques can be implemented by nesting series and parallel processor groups.
# Keeping this restriction (at least for now) means the graph structure can be represented and serialized
# simply as a list, each element of which is a processor or a list.
# This also simplifies the UI since the "graph" connectivity is implicit in the positional order.
# Thus, we don't need to support arbitrary connectivity, and can instead use a simple drag-and-drop UX paradigm.


def tick_buffer_series(carry, X, processor_names):
    params, state = carry

    Y = X
    for i, processor_name in enumerate(processor_names):
        processor_state = state[i]
        processor_params = params[i]
        processor_carry, Y = processor_by_name[processor_name].tick_buffer(
            (processor_params, processor_state), Y
        )
        state[i] = processor_carry[1]

    return carry, Y


def tick_buffer_parallel(carry, X, processor_names):
    params, state = carry

    Y = jnp.zeros(X.shape)
    for i, processor_name in enumerate(processor_names):
        processor_state = state[i]
        processor_params = params[i]
        processor_carry, Y_i = processor_by_name[processor_name].tick_buffer(
            (processor_params, processor_state), X
        )
        Y += Y_i
        state[i] = processor_carry[1]

    return carry, Y


# `processor_names`, and the params/state tuples in `carry`, are each lists,
# each element of which can be a processor or a list.
# The top-level list is interpreted as a series-connected chain,
# and every nested list is interpreted as a parallel-connected chain.
# E.g. `[["Sine Wave", "Sine Wave"], "Allpass Filter"]`
# is two parallel sine wave processors followed by an allpass filter.
def tick_buffer(carry, X, processor_names):
    params, state = carry

    Y = X
    for i, processor_name in enumerate(processor_names):
        processor_state = state[i]
        processor_params = params[i]
        processor_carry, Y = (
            tick_buffer_parallel((processor_params, processor_state), Y, processor_name)
            if isinstance(processor_name, list)
            else processor_by_name[processor_name].tick_buffer(
                (processor_params, processor_state), Y
            )
        )
        state[i] = processor_carry[1]

    return carry, Y
