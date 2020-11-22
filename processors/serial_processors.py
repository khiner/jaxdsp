import jax.numpy as jnp
from scipy import signal

NAME = 'serial_processors'

class SerialProcessors():
    def __init__(self, processor_classes):
        self.processors = [processor_class() for processor_class in processor_classes]
        self.init_params = {processor.NAME: processor.init_params for processor in self.processors}

    def create_params_target(self):
        return {processor.__class__.NAME: processor.create_params_target() for processor in self.processors}

    def tick(self, x, params):
        y = x
        for processor in self.processors:
            y = processor.tick(y, params[processor.__class__.NAME])
        return y
