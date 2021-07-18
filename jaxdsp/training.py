from jaxdsp.processors.base import processor_triplet_for_config
import numpy as np
from jax import grad, value_and_grad, jit
from jax.tree_util import tree_map, tree_multimap

from jaxdsp.processors import params_to_unit_scale, params_from_unit_scale, default_param_values
from jaxdsp.loss import LossOptions, loss_fn
from jaxdsp.optimizers import create_optimizer


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
        processor_config=None,
        loss_options=None,
        optimizer_options=None,
        track_history=False,
    ):
        self.step_num = 0
        self.loss = 0.0
        self.track_history = track_history
        self.processor = None
        self.current_params = None
        self.processor_names = None
        self.set_optimizer_options(optimizer_options)
        self.set_processor_config(processor_config)
        self.set_loss_options(loss_options)

    def set_processor_config(self, processor_config):
        self.processor_names = [processor["name"] for processor in processor_config] if processor_config else None
        self.set_processor(*processor_triplet_for_config(processor_config))

    def set_processor(self, processor, params=None, state=None):
        self.processor = processor
        if processor:
            self.processor_state = state or processor.init_state()
            self.current_params = params or default_param_values(processor, self.processor_names)
            self.step_evaluator = LossHistoryAccumulator(self.current_params)
        else:
            self.processor_state = None
            self.current_params = None
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

        if self.processor:
            self.processor_state = self.processor.init_state()

        self.update_opt_state()

    def update_opt_state(self):
        self.opt_state = self.optimizer.init(params_to_unit_scale(self.current_params, self.processor_names)) if self.current_params and self.processor else None

    def set_loss_options(self, loss_options):
        self.loss_options = loss_options or LossOptions()

        def processor_loss(unit_scale_params, state, X, Y_target):
            params = params_from_unit_scale(unit_scale_params, self.processor_names)
            carry, Y_estimated = self.processor.tick_buffer((params, state), X, self.processor_names)
            if Y_estimated.shape == Y_target.shape[::-1]:
                Y_estimated = Y_estimated.T  # TODO should eventually remove this check
            return (
                loss_fn(Y_estimated, Y_target, self.loss_options),
                carry[1], # return state as aux
            )

        self.grad_fn = jit(value_and_grad(processor_loss, has_aux=True))

    def step(self, X, Y_target):
        params_unit = params_to_unit_scale(self.current_params, self.processor_names)
        (self.loss, self.processor_state), self.grads = self.grad_fn(
            params_unit,
            self.processor_state,
            X,
            Y_target,
        )
        self.opt_state = self.optimizer.update(
            self.step_num, self.grads, self.opt_state
        )
        self.step_num += 1
        self.current_params = params_from_unit_scale(
            self.optimizer.get_params(self.opt_state), self.processor_names
        )
        if self.step_evaluator:
            self.step_evaluator.after_step(self.loss, self.current_params)

    def params(self):
        return float_params(self.current_params)

    def params_and_loss(self):
        return {
            "params": self.params(),
            "loss": float(self.loss),
        }


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target
