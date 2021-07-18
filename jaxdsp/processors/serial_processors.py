from jaxdsp.processors import processor_by_name

NAME = "Serial Processors"
PARAMS = []
PRESETS = {}

def tick_buffer(carry, X, processor_names):
    state = carry["state"]
    params = carry["params"]
    assert len(state) == len(params)

    Y = X
    for i in range(len(state)):
        processor_state = state[i]
        processor_params = params[i]
        processor = processor_by_name[processor_names[i]]
        processor_carry = {"state": processor_state, "params": processor_params}
        processor_carry, Y = processor.tick_buffer(processor_carry, Y)
        state[i] = processor_carry["state"]
        params[i] = processor_carry["params"]
    return carry, Y
