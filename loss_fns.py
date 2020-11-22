def mse(X, Y):
    return ((Y - X) ** 2).mean()

def mae(X, Y):
    return jnp.abs(Y - X).mean()

def processor_loss(params, process, processor, X, Y, Y_target):
    Y = process(params, processor, X, Y)
    return mse(Y, Y_target)

def processor_loss_serial(params, process, processor, processors, X, Y, Y_target):
    Y = process(params, processor, processors, X, Y)
    return mse(Y, Y_target)
