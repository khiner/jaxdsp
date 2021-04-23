import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.optimizers import optimizer
from jax.tree_util import tree_map

from jaxdsp.param import Param

step_size_param = Param("step_size", 0.05, 1e-9, 0.2)

# Clip all params to [0.0, 1.0] (params are all normalized to unit scale before passing to gradient fn).
# TODO should also add a loss component to strongly encourage params into this range
@optimizer
def param_clipping_optimizer(init, update, get_params):
    # Note that these are the params being optimized, NOT the optimizer params :)
    def get_clipped_params(state):
        params = get_params(state)
        return tree_map(lambda param: jnp.clip(param, 0.0, 1.0), params)

    return init, update, get_clipped_params


class Optimizer:
    def __init__(self, definition, param_values=None):
        self.definition = definition
        self.set_param_values(param_values)

    def set_param_values(self, param_values=None):
        self.param_values = {
            param.name: (param_values or {}).get(param.name) or param.default_value
            for param in self.definition.PARAMS
        }
        init, update, get_params = self.definition.FUNCTION(
            *[self.param_values[param.name] for param in self.definition.PARAMS]
        )
        self.init, self.update, self.get_params = param_clipping_optimizer(
            init, update, get_params
        )

    def serialize(self):
        return {
            "name": self.definition.NAME,
            "param_definitions": [
                param.serialize() for param in self.definition.PARAMS
            ],
            "params": self.param_values,
        }


class AdaGrad:
    NAME = "AdaGrad"
    PARAMS = [step_size_param, Param("momentum", 0.9)]
    FUNCTION = optimizers.adagrad


class Adam:
    NAME = "Adam"
    PARAMS = [step_size_param, Param("b1", 0.9), Param("b2", 0.999)]
    FUNCTION = optimizers.adam


class Adamax:
    NAME = "Adamax"
    PARAMS = [step_size_param, Param("b1", 0.9), Param("b2", 0.999)]
    FUNCTION = optimizers.adamax


class Nesterov:
    NAME = "Nesterov"
    PARAMS = [step_size_param, Param("mass", 0.5)]
    FUNCTION = optimizers.nesterov


class RmsProp:
    NAME = "RMSProp"
    PARAMS = [step_size_param, Param("gamma", 0.9)]
    FUNCTION = optimizers.rmsprop


class Sgd:
    NAME = "SGD"
    PARAMS = [step_size_param]
    FUNCTION = optimizers.sgd


all_optimizer_definitions = [AdaGrad, Adam, Adamax, Nesterov, RmsProp, Sgd]


def create_optimizer(name=None, param_values=None):
    definition = next(
        (
            definition
            for definition in all_optimizer_definitions
            if definition.NAME == name
        ),
        RmsProp,
    )

    return Optimizer(definition, param_values)
