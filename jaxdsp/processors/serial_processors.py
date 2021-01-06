NAME = 'Serial Processors'

def init_state(processors):
    return {processor.NAME: { 'state': processor.init_state(), 'tick_buffer': processor.tick_buffer } for processor in processors}

def init_params(processors):
    return {processor.NAME: processor.init_params() for processor in processors}

def default_target_params(processors):
    return {processor.NAME: processor.default_target_params() for processor in processors}

def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']
    Y = X
    # XXX this quite right. need to clean up `carry` API
    for processor_name in state.keys():
        processor_carry = {'state': state[processor_name]['state'], 'params': params[processor_name]}
        processor_carry, Y = state[processor_name]['tick_buffer'](processor_carry, Y)
        # state[processor_name] = processor_carry['state']
        # params[processor_name] = processor_carry['params']
    return carry, Y
