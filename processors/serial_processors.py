import jax.numpy as jnp
from scipy import signal

NAME = 'serial_processors'

class SerialProcessors():
    def init_params(processor_classes):
        return {processor_class.NAME: processor_class.init_params() for processor_class in processor_classes}

    def create_params_target(processor_classes):
        return {processor_class.NAME: processor_class.create_params_target() for processor_class in processor_classes}


    def __init__(self, processor_classes):
        self.processors = [processor_class() for processor_class in processor_classes]

    def tick(self, x, params):
        y = x
        for processor in self.processors:
            y = processor.tick(y, params[processor.__class__.NAME])
        return y
