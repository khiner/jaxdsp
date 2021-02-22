from jaxdsp.processors.base import Config

from jaxdsp.processors import lowpass_feedback_comb_filter as lbcf

NAME = "Serial Processors"
PARAMS = []
PRESETS = {}


def config(processors):
    configs = {processor.NAME: processor.config() for processor in processors}
    return Config(
        {
            processor.NAME: {
                "state": configs[processor.NAME].state_init,
                # "tick_buffer": processor.tick_buffer,
            }
            for processor in processors
        },
        {
            processor.NAME: configs[processor.NAME].params_init
            for processor in processors
        },
        {
            processor.NAME: configs[processor.NAME].params_target
            for processor in processors
        },
        " + ".join(processor.NAME for processor in processors),
    )


def tick_buffer(carry, X):
    state = carry["state"]
    params = carry["params"]
    Y = X
    for processor_name in state.keys():
        processor_carry = {
            "state": state[processor_name]["state"],
            "params": params[processor_name],
        }
        processor_carry, Y = lbcf.tick_buffer(processor_carry, Y)
        # state[processor_name]["state"] = processor_carry['state']
        # params[processor_name] = processor_carry['params']
    return carry, Y
