import jax.numpy as jnp
from scipy import signal

NAME = 'serial_processors'

def create_params_target(processors):
    return {processor.NAME: processor.create_params_target() for processor in processors}

def init_params(processors):
    return {processor.NAME: processor.init_params() for processor in processors}

def init_state_from_params(processors, params):
    return {processor.NAME: processor.init_state_from_params(params[processor.NAME]) for processor in processors}

def tick(processors, x, params, state):
    y = x
    for processor in processors:
        y, processor_state = processor.tick(y, params[processor.NAME], state[processor.NAME])
        state[processor.NAME] = processor_state
    return y, state
