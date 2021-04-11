from jax.experimental import optimizers
from jax.experimental.optimizers import optimizer
from jax.tree_util import tree_map

from jaxdsp.param import Param

step_size_param = Param("step_size", 0.05, 1e-9, 0.2)

# Clip all params to [0.0, 1.0] (params are all normalized to unit scale before passing to gradient fn).
# TODO should also add a loss component to strongly encourage params into this range
@optimizer
def param_clipping_optimizer(init, update, get_params):
    def get_clipped_params(state):
        params = get_params(state)
        return tree_map(lambda param: max(0.0, min(param, 1.0)), params)

    return init, update, get_clipped_params


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


all_optimizers = [AdaGrad, Adam, Adamax, Nesterov, RmsProp, Sgd]


class OptimizationOptions:
    def __init__(self, options_dict={}):
        self.options = options_dict

    def create(self):
        optimizer_name = self.options.get("name") or RmsProp.NAME
        params = self.options.get("params") or {}
        optimizer = next(
            (
                optimizer
                for optimizer in all_optimizers
                if optimizer.NAME == optimizer_name
            ),
            None,
        )
        init, update, get_params = optimizer.FUNCTION(
            *[
                params.get(param.name) or param.default_value
                for param in optimizer.PARAMS
            ]
        )
        return param_clipping_optimizer(init, update, get_params)


default_optimization_options = OptimizationOptions()
