import sys
sys.path.append('./')
sys.path.append('./processors')
from serial_processors import SerialProcessors
from process import process, process_serial

def mse(X, Y):
    return ((Y - X) ** 2).mean()

def mae(X, Y):
    return jnp.abs(Y - X).mean()

def processor_loss(params, processor_class, X, Y, Y_target):
    Y = process(params, processor_class, X, Y)
    return mse(Y, Y_target)

def processor_loss_serial(params, processor_class, processor_classes, X, Y, Y_target):
    Y = process_serial(params, processor_class, processor_classes, X, Y)
    return mse(Y, Y_target)
