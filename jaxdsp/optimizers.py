from jax.experimental import optimizers

from jaxdsp.param import Param

step_size_param = Param("step_size", 0.1, 1e-9, 4)


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
    def __init__(self, options_dict):
        self.options = options_dict

    def create(self):
        optimizer_name = self.options["name"] or Sgd.NAME
        params = self.options["params"]
        optimizer = next(
            (
                optimizer
                for optimizer in all_optimizers
                if optimizer.NAME == optimizer_name
            ),
            None,
        )
        return optimizer.FUNCTION(
            *[
                params.get(param.name) or param.default_value
                for param in optimizer.PARAMS
            ]
        )


default_optimization_options = OptimizationOptions(
    {"name": Sgd.NAME, "params": {step_size_param.name: 0.2}}
)
