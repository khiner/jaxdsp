def process(params, processor, X, Y):
    state = processor.init_state_from_params(params)
    for i, x in enumerate(X):
        y, state = processor.tick(x, params, state)
        Y = Y.at[i].set(y) # Y[i] = y
    return Y

def process_serial(params, processor, processors, X, Y):
    state = processor.init_state_from_params(processors, params)
    for i, x in enumerate(X):
        y, state = processor.tick(processors, x, params, state)
        Y = Y.at[i].set(y) # Y[i] = y
    return Y
