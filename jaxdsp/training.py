import numpy as np
from jax import value_and_grad, jit
from jax.tree_util import tree_map, tree_multimap

from jaxdsp import processor_graph
from jaxdsp.loss import LossOptions, loss_fn
from jaxdsp.optimizers import create_optimizer
from jaxdsp.processors import (
    processor_names_from_graph_config,
    processor_by_name,
    graph_config_to_carry,
    params_to_unit_scale,
    params_from_unit_scale,
    get_graph_params,
    processor_names_to_graph_config,
)


@jit
def mean_loss_and_grads(loss, grads):
    return np.mean(loss), tree_map(lambda grad: np.mean(grad, axis=0), grads)


class LossHistoryAccumulator:
    def __init__(self, params_init):
        self.params_history = tree_map(lambda param: [param], params_init)
        self.loss_history = []

    def after_step(self, loss, new_params):
        self.loss_history.append(loss)
        self.params_history = tree_multimap(
            lambda new_param, params: params + [new_param],
            new_params,
            self.params_history,
        )


def float_params(params):
    return tree_map(
        lambda param: param
        if hasattr(param, "ndim") and param.ndim > 0
        else float(param),
        params,
    )


class IterativeTrainer:
    def __init__(
        self,
        graph_config=None,
        loss_options=None,
        optimizer_options=None,
        track_history=False,
    ):
        self.step_num = 0
        self.loss = 0.0
        self.track_history = track_history
        self.params = None
        self.processor_names = None
        self.set_optimizer_options(optimizer_options)
        self.set_graph_config(graph_config)
        self.set_loss_options(loss_options)

    def set_graph_config(self, graph_config):
        self.processor_names = processor_names_from_graph_config(graph_config)
        self.set_carry(graph_config_to_carry(graph_config))

    def set_carry(self, carry):
        if carry:
            params, self.state = carry
            self.params = params or get_graph_params(
                processor_names_to_graph_config(self.processor_names)
            )
            self.step_evaluator = LossHistoryAccumulator(self.params)
        else:
            self.params, self.state = None, None
            self.step_evaluator = None

        self.update_opt_state()

    def set_optimizer_options(self, optimizer_options):
        self.optimizer = (
            create_optimizer(
                optimizer_options.get("name"), optimizer_options.get("params")
            )
            if optimizer_options
            else create_optimizer()
        )

        if self.processor_names:
            self.state = [
                processor_by_name[processor_name].init_state()
                for processor_name in self.processor_names
            ]

        self.update_opt_state()

    def update_opt_state(self):
        self.opt_state = self.optimizer.init(
            params_to_unit_scale(self.params, self.processor_names)
        )

    def set_loss_options(self, loss_options):
        self.loss_options = loss_options or LossOptions()

        def processor_loss(unit_scale_params, state, X, Y_target):
            params = params_from_unit_scale(unit_scale_params, self.processor_names)
            carry, Y_estimated = processor_graph.tick_buffer(
                (params, state), X, self.processor_names
            )
            if Y_estimated.shape == Y_target.shape[::-1]:
                Y_estimated = Y_estimated.T  # TODO should eventually remove this check
            return (
                loss_fn(Y_estimated, Y_target, self.loss_options),
                carry[1],  # return state as aux
            )

        self.grad_fn = jit(value_and_grad(processor_loss, has_aux=True))

    def step(self, X, Y_target):
        if not self.processor_names:
            return

        params_unit = params_to_unit_scale(self.params, self.processor_names)
        (self.loss, self.state), self.grads = self.grad_fn(
            params_unit,
            self.state,
            X,
            Y_target,
        )
        self.opt_state = self.optimizer.update(
            self.step_num, self.grads, self.opt_state
        )
        self.step_num += 1
        self.params = params_from_unit_scale(
            self.optimizer.get_params(self.opt_state), self.processor_names
        )
        if self.step_evaluator:
            self.step_evaluator.after_step(self.loss, self.params)

    def float_params(self):
        return float_params(self.params)

    def params_and_loss(self):
        return {
            "params": self.float_params(),
            "loss": float(self.loss),
        }


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target
