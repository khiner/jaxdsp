def process(params, processor_class, X, Y):
    processor = processor_class()
    for i, x in enumerate(X):
        y = processor.tick(x, params)
        Y = Y.at[i].set(y) # Y[i] = y
    return Y

def process_serial(params, processor_class, processors, X, Y):
    processor = processor_class(processors)
    for i, x in enumerate(X):
        y = processor.tick(x, params)
        Y = Y.at[i].set(y) # Y[i] = y
    return Y
