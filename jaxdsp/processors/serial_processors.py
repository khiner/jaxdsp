from jaxdsp.processors.base import Config
from jaxdsp.processors import processor_by_name

NAME = "Serial Processors"
PARAMS = []
PRESETS = {}


def config(processors):
    return Config(
        {processor.NAME: processor.config().state_init for processor in processors},
        " + ".join(processor.NAME for processor in processors),
    )


def tick_buffer(carry, X):
    state = carry["state"]
    params = carry["params"]
    Y = X
    for processor_name in state.keys():
        processor_carry = {
            "state": state[processor_name],
            "params": params[processor_name],
        }
        processor_carry, Y = processor_by_name[processor_name].tick_buffer(
            processor_carry, Y
        )
        state[processor_name] = processor_carry["state"]
        params[processor_name] = processor_carry["params"]
    return carry, Y
