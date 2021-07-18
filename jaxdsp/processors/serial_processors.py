from jaxdsp.processors import processor_by_name

NAME = "Serial Processors"
PARAMS = []
PRESETS = {}

def tick_buffer(carry, X, processor_names):
    params, state = carry
    assert len(state) == len(params)

    Y = X
    for i in range(len(state)):
        processor_state = state[i]
        processor_params = params[i]
        processor = processor_by_name[processor_names[i]]
        processor_carry, Y = processor.tick_buffer((processor_params, processor_state), Y)
        carry[0][i] = processor_carry[0]
        carry[1][i] = processor_carry[1]
    return carry, Y
